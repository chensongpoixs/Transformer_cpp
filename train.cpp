#include "train.h"
#include "beam_search.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include <iomanip>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <limits>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;
using namespace logging;

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

    if (is_training) {
        LOG_INFO("使用长度bucket采样: bucket_size=" + std::to_string(bucket_size) +
                 ", 总样本数=" + std::to_string(indices.size()));
    }

    // 按批次处理数据
    size_t num_batches = (indices.size() + batch_size - 1) / batch_size;
    {
        std::ostringstream oss;
        oss << (is_training ? "[Train] " : "[Eval] ")
            << "共 " << num_batches << " 个 batch, batch_size=" << batch_size
            << ", 样本数=" << dataset.size();
        LOG_INFO(oss.str());
    }
    
    for (size_t i = 0; i < num_batches; ++i) {
        size_t start_idx = i * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, indices.size());
        
        // 获取当前批次的索引
        std::vector<size_t> batch_indices(indices.begin() + start_idx, 
                                         indices.begin() + end_idx);
        
        // 创建batch
        auto batch = dataset.collate_fn(batch_indices, device,
                                       config.padding_idx, config.bos_idx, config.eos_idx,
                                       config.src_vocab_size, config.tgt_vocab_size);
        
        // 前向传播
        auto out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
        {
            //std::ostringstream oss;
            //oss << ""
            //LOG_DEBUG("");
        }
        
        // 计算损失
        float loss = loss_compute(out, batch.trg_y, static_cast<float>(batch.ntokens));
        
        // 累加
        total_loss += loss * batch.ntokens;
        total_tokens += batch.ntokens;
        
        // YOLOv5 风格的进度日志（每若干 batch 打一次）
        if ((i + 1) % 10 == 0 || i == num_batches - 1) {
            float progress = static_cast<float>(i + 1) / static_cast<float>(num_batches);
            int pct = static_cast<int>(progress * 100.0f + 0.5f);

            float avg_loss_so_far = (total_tokens > 0.0f)
                ? (total_loss / total_tokens)
                : 0.0f;

            std::ostringstream oss;
            oss << (is_training ? "Train" : "Val")
                << " | Epoch " << epoch << "/" << total_epochs
                << " | Batch " << (i + 1) << "/" << num_batches
                << " (" << std::setw(3) << pct << "%)"
                << " | Loss " << std::fixed << std::setprecision(3) << loss
                << " | AvgLoss " << std::fixed << std::setprecision(3) << avg_loss_so_far
                << " | Tokens " << static_cast<long long>(total_tokens);
            LOG_INFO(oss.str());
        }
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
        LOG_INFO("开始训练阶段...");
        float train_loss = run_epoch(train_dataset, model, loss_compute_train,
                                    config.batch_size, device, config, true,
                                    epoch, config.epoch_num);
        
        // 验证阶段
        model->eval();
        LOG_INFO("开始验证阶段...");
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
    // 加载中文分词器用于解码
    auto sp_chn = chinese_tokenizer_load();
    
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

