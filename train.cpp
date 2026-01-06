#include "train.h"
#include "beam_search.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include "gpu_profiler.h"
#include <iomanip>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <limits>
#include <sstream>
#include <utility>
#include <chrono>
#include <cmath>

namespace fs = std::filesystem;
using namespace logging;
using namespace std::chrono;

// 仿照 Python tools/create_exp_folder.py 的实验目录创建逻辑
// 返回: (exp_folder, weights_folder)
// 支持 YOLOv5 风格的 --project 和 --name 参数
static std::pair<std::string, std::string> create_exp_folder_cpp(
    const std::string& project,
    const std::string& name,
    bool exist_ok) {
    
    fs::path project_path(project);
    
    // 确保项目目录存在
    std::error_code ec;
    fs::create_directories(project_path, ec);
    if (ec) {
        LOG_WARN("创建项目目录失败: " + project_path.string() + ", 错误: " + ec.message());
    }

    // 首先尝试 project/name
    fs::path exp_dir = project_path / name;
    if (!fs::exists(exp_dir) || exist_ok) {
        if (exist_ok && fs::exists(exp_dir)) {
            LOG_INFO("实验目录已存在，使用现有目录: " + exp_dir.string());
        }
        fs::create_directories(exp_dir / "weights", ec);
        if (ec) {
            LOG_WARN("创建权重目录失败: " + (exp_dir / "weights").string() + ", 错误: " + ec.message());
        }
        return {exp_dir.string(), (exp_dir / "weights").string()};
    }

    // 如果 name 已存在且 exist_ok=false，按 name1, name2, ... 递增
    int exp_num = 1;
    while (true) {
        fs::path exp_dir_i = project_path / (name + std::to_string(exp_num));
        if (!fs::exists(exp_dir_i)) {
            fs::create_directories(exp_dir_i / "weights", ec);
            if (ec) {
                LOG_WARN("创建权重目录失败: " + (exp_dir_i / "weights").string() + ", 错误: " + ec.message());
            }
            return {exp_dir_i.string(), (exp_dir_i / "weights").string()};
        }
        ++exp_num;
    }
}

/**
 * YOLOv5 风格的进度条显示
 * @param epoch 当前 epoch
 * @param total_epochs 总 epoch 数
 * @param batch_idx 当前 batch 索引（从0开始）
 * @param total_batches 总 batch 数
 * @param loss 当前 batch 的损失
 * @param avg_loss 平均损失
 * @param speed 处理速度（samples/s）
 * @param eta 预计剩余时间（秒）
 * @param is_training 是否为训练模式
 * @param device GPU设备（用于显示显存使用）
 */
static void print_progress_bar(int epoch, int total_epochs,
                               size_t batch_idx, size_t total_batches,
                               float loss, float avg_loss,
                               double speed, double eta,
                               bool is_training,
                               torch::Device device) {
    const int bar_width = 30;
    float progress = static_cast<float>(batch_idx + 1) / static_cast<float>(total_batches);
    int filled = static_cast<int>(progress * bar_width);
    int pct = static_cast<int>(progress * 100.0f + 0.5f);
    
    std::string mode = is_training ? "Train" : "Val";
    std::string bar(filled, '=');
    if (filled < bar_width) {
        bar += '>';
        bar += std::string(bar_width - filled - 1, ' ');
    }
    
    // 格式化时间
    auto format_time = [](double seconds) -> std::string {
        if (seconds < 0) return "?";
        int h = static_cast<int>(seconds / 3600);
        int m = static_cast<int>((seconds - h * 3600) / 60);
        int s = static_cast<int>(seconds - h * 3600 - m * 60);
        if (h > 0) {
            return std::to_string(h) + "h" + std::to_string(m) + "m" + std::to_string(s) + "s";
        } else if (m > 0) {
            return std::to_string(m) + "m" + std::to_string(s) + "s";
        } else {
            return std::to_string(s) + "s";
        }
    };
    
    // 获取GPU内存使用情况
    std::string gpu_mem = GPUProfiler::get_gpu_memory_str(device);
    
    std::ostringstream oss;
    oss << mode << ": " << epoch << "/" << total_epochs
        << " [" << bar << "] " << std::setw(3) << pct << "%"
        << " " << std::setw(4) << (batch_idx + 1) << "/" << total_batches
        << " GPU=" << gpu_mem
        << " loss=" << std::fixed << std::setprecision(3) << loss
        << " avg_loss=" << std::fixed << std::setprecision(3) << avg_loss
        << " " << std::fixed << std::setprecision(1) << speed << " samples/s"
        << " ETA=" << format_time(eta);
    
    std::string progress_str = oss.str();
    // 添加空格以清除之前可能更长的行内容（最多120个字符）
    if (progress_str.length() < 120) {
        progress_str += std::string(120 - progress_str.length(), ' ');
    }
    
    // 使用 \r 覆盖同一行（Windows 和 Linux 都支持）
    std::cout << "\r" << progress_str << std::flush;
    
    // 如果是最后一个 batch，换行
    if (batch_idx + 1 == total_batches) {
        std::cout << std::endl;
    }
}

float run_epoch(MTDataset& dataset,
                Transformer model,
                LossCompute& loss_compute,
                int batch_size,
                torch::Device device,
                const TransformerConfig& config,
                bool is_training,
                int epoch,
                int total_epochs) {
    
    float total_tokens = 0.0f;
    float total_loss = 0.0f;
    
    // 基于句子长度的 bucket 采样策略
    // 1. 先按长度排序得到索引
    std::vector<size_t> base_indices = dataset.make_length_sorted_indices();

    // 2. 按 bucket 切分，再在 bucket 内部打乱
    std::vector<size_t> indices;
    indices.reserve(base_indices.size());

    const size_t bucket_size = static_cast<size_t>(batch_size) * 4;  // 可调：4 倍batch
    std::vector<size_t> bucket;
    bucket.reserve(bucket_size);

    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t idx : base_indices) {
        bucket.push_back(idx);
        if (bucket.size() >= bucket_size) {
            // 打乱 bucket 内部的顺序
            std::shuffle(bucket.begin(), bucket.end(), g);
            indices.insert(indices.end(), bucket.begin(), bucket.end());
            bucket.clear();
        }
    }
    // 处理最后一个不满的 bucket
    if (!bucket.empty()) {
        std::shuffle(bucket.begin(), bucket.end(), g);
        indices.insert(indices.end(), bucket.begin(), bucket.end());
    }

    // 按批次处理数据
    size_t num_batches = (indices.size() + batch_size - 1) / batch_size;
    
    if (is_training) {
        LOG_INFO("使用长度bucket采样: bucket_size=" + std::to_string(bucket_size) +
                 ", 总样本数=" + std::to_string(indices.size()) + ", 批次数=" + std::to_string(num_batches));
    }
    
    // 计时相关
    auto epoch_start = steady_clock::now();
    auto batch_start = steady_clock::now();
    size_t processed_samples = 0;
    
    for (size_t i = 0; i < num_batches; ++i) {
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, indices.size());
        
        // 获取当前批次的索引
        std::vector<size_t> batch_indices(indices.begin() + start_idx, 
                                         indices.begin() + end_idx);
        
        // 创建batch（性能分析）
        GPUProfiler::start_timer("collate_fn");
        auto batch = dataset.collate_fn(batch_indices, device,
                                       config.padding_idx, config.bos_idx, config.eos_idx,
                                       config.src_vocab_size, config.tgt_vocab_size);
        GPUProfiler::end_timer("collate_fn");
        
        // 前向传播（性能分析）
        GPUProfiler::start_timer("forward");
        auto out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
        GPUProfiler::end_timer("forward");
        
        // 计算损失（性能分析）
        GPUProfiler::start_timer("loss_compute");
        float loss = loss_compute(out, batch.trg_y, static_cast<float>(batch.ntokens));
        GPUProfiler::end_timer("loss_compute");
        
        // 累加
        total_loss += loss * batch.ntokens;
        total_tokens += batch.ntokens;
        processed_samples += batch_indices.size();
        
        // 计算速度和剩余时间（使用从 epoch 开始的总时间）
        auto batch_end = steady_clock::now();
        auto total_elapsed = duration_cast<milliseconds>(batch_end - epoch_start).count() / 1000.0;
        double speed = (total_elapsed > 0.0) ? (processed_samples / total_elapsed) : 0.0;
        
        // 计算平均损失
        float avg_loss_so_far = (total_tokens > 0.0f)
            ? (total_loss / total_tokens)
            : 0.0f;
        
        // 计算剩余时间（ETA）
        double eta = 0.0;
        if (speed > 0.0 && i + 1 < num_batches) {
            size_t remaining_samples = (num_batches - i - 1) * batch_size;
            eta = remaining_samples / speed;
        }
        
        // 显示 YOLOv5 风格的进度条（每个 batch 都更新，包含GPU内存信息）
        print_progress_bar(epoch, total_epochs, i, num_batches,
                          loss, avg_loss_so_far, speed, eta, is_training, device);
    }
    
    // 性能分析：在第一个epoch结束后打印
    if (epoch == 1 && is_training) {
        GPUProfiler::print_summary();
        GPUProfiler::check_gpu_utilization(device);
    }
    
    float avg_loss = (total_tokens > 0.0f) ? (total_loss / total_tokens) : 0.0f;
    /*{
        std::ostringstream oss;
        oss << (is_training ? "[Train] " : "[Eval] ")
            << "Epoch结束, 平均损失=" << std::fixed << std::setprecision(4) << avg_loss
            << ", 总token数=" << static_cast<long long>(total_tokens);
        LOG_INFO(oss.str());
    }*/
    return avg_loss;
}

void train(MTDataset& train_dataset,
           MTDataset& dev_dataset,
           Transformer model,
           torch::nn::CrossEntropyLoss criterion,
           std::shared_ptr<NoamOpt> optimizer,
           const TransformerConfig& config,
           torch::Device device) {
    
    // 创建实验文件夹（对齐 Python 版 create_exp_folder，支持 YOLOv5 风格）
    auto [exp_folder, weights_folder] = create_exp_folder_cpp(
        config.project, config.name, config.exist_ok);
    LOG_INFO("项目目录: " + config.project);
    LOG_INFO("实验名称: " + config.name);
    LOG_INFO("实验目录: " + exp_folder);
    LOG_INFO("权重目录: " + weights_folder);
    
    // YOLOv5 风格：基于验证损失保存最佳模型
    float best_val_loss = std::numeric_limits<float>::infinity();  // 最小验证损失
    std::string best_path = weights_folder + "/best.pth";
    std::string last_path = weights_folder + "/last.pth";
    
    // 创建损失计算器
    auto loss_compute_train = LossCompute(model->get_generator(), criterion, optimizer);
    auto loss_compute_eval = LossCompute(model->get_generator(), criterion, nullptr);
    LOG_INFO("LossCompute 对象创建完成（train & eval）");
    
    // 训练循环
    for (int epoch = 1; epoch <= config.epoch_num; ++epoch) {
        {
            std::ostringstream oss;
            oss << "========== Epoch " << epoch << "/" << config.epoch_num << " ==========";
            LOG_INFO(oss.str());
        }
        
        // 训练阶段
        model->train();
        {
            std::ostringstream oss;
            oss << "开始训练阶段...";
            if (device.is_cuda()) {
                oss << " GPU显存: " << GPUProfiler::get_gpu_memory_str(device);
            }
            LOG_INFO(oss.str());
        }
        float train_loss = run_epoch(train_dataset, model, loss_compute_train,
                                    config.batch_size, device, config, true,
                                    epoch, config.epoch_num);
        
        // 验证阶段
        model->eval();
        {
            std::ostringstream oss;
            oss << "开始验证阶段...";
            if (device.is_cuda()) {
                oss << " GPU显存: " << GPUProfiler::get_gpu_memory_str(device);
            }
            LOG_INFO(oss.str());
        }
        float dev_loss = run_epoch(dev_dataset, model, loss_compute_eval,
                                  config.batch_size, device, config, false,
                                  epoch, config.epoch_num);
        
        // 计算BLEU分数（用于监控，但不用于保存模型）
        float bleu_score = evaluate(dev_dataset, model, config, device);
        
        {
            std::ostringstream oss;
            oss << "Epoch " << epoch
                << " - TrainLoss=" << std::fixed << std::setprecision(3) << train_loss
                << ", ValLoss=" << std::fixed << std::setprecision(3) << dev_loss
                << ", BLEU=" << std::fixed << std::setprecision(2) << bleu_score;
            LOG_INFO(oss.str());
        }
        
        // YOLOv5 风格：基于验证损失保存最佳模型
        // 如果当前验证损失小于历史最小损失，保存为 best.pth
        if (dev_loss < best_val_loss) {
            try {
                torch::save(model, best_path);
                {
                    std::ostringstream oss;
                    if (best_val_loss == std::numeric_limits<float>::infinity()) {
                        oss << "保存最佳模型: " << best_path 
                            << " (ValLoss=" << std::fixed << std::setprecision(3) << dev_loss << ")";
                    } else {
                        oss << "保存最佳模型: " << best_path 
                            << " (ValLoss=" << std::fixed << std::setprecision(3) << dev_loss
                            << " < " << std::fixed << std::setprecision(3) << best_val_loss << ")";
                    }
                    LOG_INFO(oss.str());
                }
                best_val_loss = dev_loss;
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("保存最佳模型失败: ") + best_path + ", 错误: " + e.what());
            }
        }
        
        // YOLOv5 风格：每个 epoch 都保存 last.pth（覆盖之前的）
        try {
            torch::save(model, last_path);
            {
                std::ostringstream oss;
                oss << "保存最后模型: " << last_path 
                    << " (Epoch " << epoch << ", ValLoss=" 
                    << std::fixed << std::setprecision(3) << dev_loss << ")";
                LOG_INFO(oss.str());
            }
        } catch (const std::exception& e) {
            LOG_ERROR(std::string("保存最后模型失败: ") + last_path + ", 错误: " + e.what());
        }
    }
    
    // 训练结束，输出总结
    {
        std::ostringstream oss;
        oss << "========== 训练完成 ==========";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "最佳验证损失: " << std::fixed << std::setprecision(3) << best_val_loss
            << " (保存在: " << best_path << ")";
        LOG_INFO(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "最后模型: " << last_path;
        LOG_INFO(oss.str());
    }
}

float evaluate(MTDataset& dataset,
               Transformer model,
               const TransformerConfig& config,
               torch::Device device) {
    // 使用配置的中文分词器路径加载分词器用于解码
    auto sp_chn = chinese_tokenizer_load(config.tokenizer_chn);
    
    model->eval();
    torch::NoGradGuard no_grad;
    
    std::vector<std::vector<std::string>> all_candidates;
    std::vector<std::vector<std::vector<std::string>>> all_references;
    
    // 评估所有数据（或限制数量）
    size_t eval_size = dataset.size();
    std::vector<size_t> indices(eval_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    for (size_t i = 0; i < indices.size(); i += config.batch_size) {
        size_t end = std::min(i + config.batch_size, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + i, indices.begin() + end);
        
        // 获取batch数据
        auto batch = dataset.collate_fn(batch_indices, device,
                                       config.padding_idx, config.bos_idx, config.eos_idx,
                                       config.src_vocab_size, config.tgt_vocab_size);
        
        // 创建src_mask
        auto src_mask = (batch.src != config.padding_idx).unsqueeze(-2);
        
        // 使用beam search解码
        auto [decode_results, scores] = beam_search(
            model,
            batch.src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            device
        );
        
        // 处理解码结果
        for (size_t j = 0; j < decode_results.size(); ++j) {
            // 取最佳结果（第一个）
            if (!decode_results[j].empty()) {
                // 将token ID转换为字符串
                std::string translation = sp_chn->decode_ids(decode_results[j][0]);
                all_candidates.push_back(tokenize_chinese(translation));
            } else {
                all_candidates.push_back({});
            }
            
            // 参考句子（真实目标文本）
            std::vector<std::vector<std::string>> refs;
            refs.push_back(tokenize_chinese(batch.trg_text[j]));
            all_references.push_back(refs);
        }
    }
    
    // 计算BLEU分数
    float bleu_score = corpus_bleu(all_candidates, all_references, 4);
    return bleu_score;
}

