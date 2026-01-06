#include "train.h"
#include "beam_search.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include "gpu_profiler.h"
#include "json.hpp"
#include <iomanip>
#include <algorithm>
#include <random>
#include <filesystem>
#include <numeric>
#include <limits>
#include <sstream>
#include <utility>
#include <tuple>
#include <chrono>
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ctime>


namespace fs = std::filesystem;
using namespace logging;
using namespace std::chrono;
using json = nlohmann::json;

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
 * 保存训练配置文件（YOLOv5 风格）
 * @param config 训练配置
 * @param exp_folder 实验文件夹路径
 */
static void save_config_file(const TransformerConfig& config, const std::string& exp_folder) {
    // 保存到文件（使用 config.yaml）
    std::string config_path = exp_folder + "/config.yaml";
    std::ofstream config_file(config_path);
    if (!config_file.is_open()) {
        LOG_WARN("无法保存训练配置: " + config_path);
        return;
    }
    
    // YOLOv5 风格的 YAML 格式，带注释和分组
    config_file << "# Transformer Training Configuration\n";
    config_file << "# Generated automatically during training\n\n";
    
    // Train 训练配置
    config_file << "# Train\n";
    config_file << "epochs: " << config.epoch_num << "  # 训练轮数\n";
    config_file << "batch_size: " << config.batch_size << "  # 批次大小\n";
    config_file << "lr: " << std::scientific << config.lr << "  # 学习率\n";
    config_file << "workers: " << config.workers << "  # 数据加载线程数\n";
    config_file << "\n";
    
    // Model 模型配置
    config_file << "# Model\n";
    config_file << "d_model: " << config.d_model << "  # 模型维度\n";
    config_file << "n_heads: " << config.n_heads << "  # 多头注意力头数\n";
    config_file << "n_layers: " << config.n_layers << "  # Transformer层数\n";
    config_file << "d_k: " << config.d_k << "  # 每个头的键向量维度\n";
    config_file << "d_v: " << config.d_v << "  # 每个头的值向量维度\n";
    config_file << "d_ff: " << config.d_ff << "  # 前馈网络隐藏层维度\n";
    config_file << "dropout: " << std::fixed << std::setprecision(2) << config.dropout << "  # Dropout率\n";
    config_file << "\n";
    
    // Vocabulary 词汇表配置
    config_file << "# Vocabulary\n";
    config_file << "src_vocab_size: " << config.src_vocab_size << "  # 源语言词汇表大小\n";
    config_file << "tgt_vocab_size: " << config.tgt_vocab_size << "  # 目标语言词汇表大小\n";
    config_file << "padding_idx: " << config.padding_idx << "  # Padding标记索引\n";
    config_file << "bos_idx: " << config.bos_idx << "  # 开始符索引\n";
    config_file << "eos_idx: " << config.eos_idx << "  # 结束符索引\n";
    config_file << "\n";
    
    // Decode 解码配置
    config_file << "# Decode\n";
    config_file << "max_len: " << config.max_len << "  # 最大序列长度\n";
    config_file << "beam_size: " << config.beam_size << "  # Beam Search大小\n";
    config_file << "\n";
    
    // Data 数据路径配置
    config_file << "# Data\n";
    config_file << "data_dir: " << config.data_dir << "  # 数据目录\n";
    config_file << "train: " << config.train_data_path << "  # 训练集路径\n";
    config_file << "val: " << config.dev_data_path << "  # 验证集路径\n";
    config_file << "test: " << config.test_data_path << "  # 测试集路径\n";
    config_file << "\n";
    
    // Tokenizer 分词器配置
    config_file << "# Tokenizer\n";
    config_file << "tokenizer_dir: " << config.tokenizer_dir << "  # 分词器目录\n";
    config_file << "tokenizer_eng: " << config.tokenizer_eng << "  # 英文分词器模型路径\n";
    config_file << "tokenizer_chn: " << config.tokenizer_chn << "  # 中文分词器模型路径\n";
    config_file << "\n";
    
    // Project 项目配置
    config_file << "# Project\n";
    config_file << "project: " << config.project << "  # 项目目录\n";
    config_file << "name: " << config.name << "  # 实验名称\n";
    config_file << "exist_ok: " << (config.exist_ok ? "true" : "false") << "  # 是否覆盖已存在目录\n";
    config_file << "\n";
    
    // Device 设备配置
    config_file << "# Device\n";
    config_file << "use_cuda: " << (config.use_cuda ? "true" : "false") << "  # 是否使用CUDA\n";
    config_file << "device_id: " << config.device_id << "  # GPU设备ID\n";
    
    config_file.close();
    LOG_INFO("保存训练配置: " + config_path);
}


/**
 * YOLOv5 风格的表格格式实时更新（带进度条）
 * 格式:   1/100     2.5G   100/20     1.5M      0.1200     0.1420    13.50    45.6s   50%|==========>          |
 */
static void print_progress_bar(int epoch, int total_epochs,
                               size_t batch_idx, size_t total_batches,
                               float loss, float avg_loss,
                               double speed, double eta,
                               bool is_training,
                               torch::Device device, double elapsed_time,
                               long long current_tokens, size_t current_batches) {
    // 计算进度条
    const int bar_width = 20;
    float progress = static_cast<float>(batch_idx + 1) / static_cast<float>(total_batches);
    int filled = static_cast<int>(progress * bar_width);
    int pct = static_cast<int>(progress * 100.0f + 0.5f);
    
    // 使用ASCII字符构建进度条
    std::string bar;
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            bar += '=';
        } else if (i == filled && filled < bar_width) {
            bar += '>';
        } else {
            bar += ' ';
        }
    }
    
    // 获取GPU内存使用情况
    std::string gpu_mem = "N/A";
    if (device.is_cuda()) {
        try {
            c10::cuda::CUDAGuard guard(device);
            size_t allocated = 0;
            size_t total = 0;
#ifdef USE_CUDA
            size_t free = 0;
            if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                allocated = total - free;
                double allocated_gb = allocated / (1024.0 * 1024.0 * 1024.0);
                std::ostringstream gpu_oss;
                gpu_oss << std::fixed << std::setprecision(1) << allocated_gb << "G";
                gpu_mem = gpu_oss.str();
            }
#endif
        } catch (...) {
            gpu_mem = "N/A";
        }
    } else {
        gpu_mem = "0G";
    }
    
    // 格式化批次数量（显示当前批次/总批次）
    std::ostringstream batch_oss;
    batch_oss << (batch_idx + 1) << "/" << total_batches;
    
    // 计算每秒处理的token数量
    double tokens_per_sec = (elapsed_time > 0.0) ? (static_cast<double>(current_tokens) / elapsed_time) : 0.0;
    
    // 格式化每秒tokens数量（使用K/M/G等单位，添加/s后缀）
    std::string tokens_str;
    if (tokens_per_sec >= 1000000000) {
        std::ostringstream t_oss;
        t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000000000.0) << "G/s";
        tokens_str = t_oss.str();
    } else if (tokens_per_sec >= 1000000) {
        std::ostringstream t_oss;
        t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000000.0) << "M/s";
        tokens_str = t_oss.str();
    } else if (tokens_per_sec >= 1000) {
        std::ostringstream t_oss;
        t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000.0) << "K/s";
        tokens_str = t_oss.str();
    } else {
        std::ostringstream t_oss;
        t_oss << std::fixed << std::setprecision(1) << tokens_per_sec << "/s";
        tokens_str = t_oss.str();
    }
    
    // YOLOv5风格：表格格式输出（与epoch汇总行格式一致）+ 进度条
    // 格式: train:  1/100      2.5G        100/20      1.5M          0.1200        -         -       45.6s        |==========>          | 50%
    // YOLOv5风格：所有列左对齐
    std::ostringstream oss;
    oss << "train: "
        << std::setw(10) << std::left << (std::to_string(epoch) + "/" + std::to_string(total_epochs))
        << std::setw(12) << std::left << gpu_mem
        << std::setw(15) << std::left << batch_oss.str()
        << std::setw(15) << std::left << tokens_str
        << std::setw(15) << std::left << std::fixed << std::setprecision(4) << avg_loss;
    
    // 训练阶段：val_loss和BLEU显示为"-"
    if (is_training) {
        oss << std::setw(15) << std::left << "-"
            << std::setw(10) << std::left << "-";
    } else {
        // 验证阶段：显示当前损失（val_loss），BLEU显示为"-"
        oss << std::setw(15) << std::left << std::fixed << std::setprecision(4) << avg_loss
            << std::setw(10) << std::left << "-";
    }
    
    oss << std::setw(10) << std::left << std::fixed << std::setprecision(1) << elapsed_time << "s"
        << std::setw(28) << std::left << ("|" + bar + "| " + std::to_string(pct) + "%");
    
    std::string progress_str = oss.str();
    
    // 添加空格以清除之前可能更长的行内容
    const int terminal_width = 140;
    if (progress_str.length() < terminal_width) {
        progress_str += std::string(terminal_width - progress_str.length(), ' ');
    }
    
    // 使用 \r 覆盖同一行
    std::cout << "\r" << progress_str << std::flush;
    
    // 如果是最后一个 batch，换行
    if (batch_idx + 1 == total_batches) {
        std::cout << std::endl;
    }
}

// 返回 (平均损失, 总tokens数, 批次数量)
std::tuple<float, long long, size_t> run_epoch(MTDataset& dataset,
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
    
    // 计时相关
    auto epoch_start = steady_clock::now();
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
        // 验证阶段使用 NoGradGuard 避免构建计算图，节省显存
        torch::Tensor out;
        if (is_training) {
            GPUProfiler::start_timer("forward");
            out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
            GPUProfiler::end_timer("forward");
        } else {
            torch::NoGradGuard no_grad;
            GPUProfiler::start_timer("forward");
            out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
            GPUProfiler::end_timer("forward");
        }
        
        // 计算损失（性能分析）
        GPUProfiler::start_timer("loss_compute");
        float loss = loss_compute(out, batch.trg_y, static_cast<float>(batch.ntokens));
        GPUProfiler::end_timer("loss_compute");
        
        // 累加
        total_loss += loss * batch.ntokens;
        total_tokens += batch.ntokens;
        processed_samples += batch_indices.size();
        
        // 显式释放中间张量（帮助释放显存）
        // 训练和验证阶段都需要释放，避免张量引用累积
        out = torch::Tensor();
        
        // 计算速度和剩余时间（使用从 epoch 开始的总时间）
        auto batch_end = steady_clock::now();
        double elapsed_time = duration_cast<milliseconds>(batch_end - epoch_start).count() / 1000.0;
        double speed = (elapsed_time > 0.0) ? (processed_samples / elapsed_time) : 0.0;
        
        // 计算平均损失
        float avg_loss_so_far = (total_tokens > 0.0f)
            ? (total_loss / total_tokens)
            : 0.0f;
        
        // 计算剩余时间（ETA）：使用剩余batch数计算更准确
        double eta = 0.0;
        if (speed > 0.0 && i + 1 < num_batches) {
            size_t remaining_batches = num_batches - i - 1;
            // 使用平均每个batch的样本数来估算剩余样本数
            double avg_samples_per_batch = static_cast<double>(processed_samples) / (i + 1);
            double remaining_samples = remaining_batches * avg_samples_per_batch;
            eta = remaining_samples / speed;
        }
        
        // 显示 YOLOv5 风格的表格格式实时更新（每个 batch 都更新）
        print_progress_bar(epoch, total_epochs, i, num_batches,
                          loss, avg_loss_so_far, speed, eta, is_training, device, elapsed_time,
                          static_cast<long long>(total_tokens), num_batches);
        
        // 定期清理CUDA缓存（每50个batch清理一次，避免频繁清理影响性能）
        if (device.is_cuda() && (i + 1) % 50 == 0) {
          //  torch::cuda::empty_cache();
        }
    }
    
    // 性能分析：在第一个epoch结束后打印
    if (epoch == 1 && is_training) {
        GPUProfiler::print_summary();
        GPUProfiler::check_gpu_utilization(device);
    }
    
    // epoch结束后清理CUDA缓存
    if (device.is_cuda()) {
       // torch::cuda::empty_cache();
       // torch::cuda::synchronize();
    }
    
    float avg_loss = (total_tokens > 0.0f) ? (total_loss / total_tokens) : 0.0f;
    long long total_tokens_long = static_cast<long long>(total_tokens);
    /*{
        std::ostringstream oss;
        oss << (is_training ? "[Train] " : "[Eval] ")
            << "Epoch结束, 平均损失=" << std::fixed << std::setprecision(4) << avg_loss
            << ", 总token数=" << total_tokens_long
            << ", 批次数=" << num_batches;
        LOG_INFO(oss.str());
    }*/
    return std::make_tuple(avg_loss, total_tokens_long, num_batches);
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
    
    // 保存训练配置文件（YOLOv5 风格）
    save_config_file(config, exp_folder);
    
    // YOLOv5 风格：基于验证损失保存最佳模型
    float best_val_loss = std::numeric_limits<float>::infinity();  // 最小验证损失
    std::string best_path = weights_folder + "/best.pth";
    std::string last_path = weights_folder + "/last.pth";
    
    // 创建损失计算器
    auto loss_compute_train = LossCompute(model->get_generator(), criterion, optimizer);
    auto loss_compute_eval = LossCompute(model->get_generator(), criterion, nullptr);
    LOG_INFO("LossCompute 对象创建完成（train & eval）");
    
    // 计算训练数据集的bucket采样信息（在训练开始前打印）
    const size_t bucket_size = static_cast<size_t>(config.batch_size) * 4;  // 可调：4 倍batch
    size_t train_dataset_size = train_dataset.size();
    size_t train_num_batches = (train_dataset_size + config.batch_size - 1) / config.batch_size;
    LOG_INFO("使用长度bucket采样: bucket_size=" + std::to_string(bucket_size) +
             ", 总样本数=" + std::to_string(train_dataset_size) + ", 批次数=" + std::to_string(train_num_batches));
    
    // YOLOv5风格：在训练开始前打印表头
    std::cout << std::endl;
    // 表头格式：train: Epoch   GPU_mem   Batch      Tokens     train_loss    val_loss     BLEU     time   进度条
    // 注意：宽度要与实际输出完全一致，进度条部分固定为28个字符（"|====================| 100%"）
    // YOLOv5风格：表头字段左对齐
    std::cout << "train: "
              << std::setw(10) << std::left << "Epoch"
              << std::setw(12) << std::left << "GPU_mem"
              << std::setw(15) << std::left << "Batch"
              << std::setw(15) << std::left << "Tokens"
              << std::setw(15) << std::left << "train_loss"
              << std::setw(15) << std::left << "val_loss"
              << std::setw(10) << std::left << "BLEU"
              << std::setw(10) << std::left << "time"
              << std::setw(28) << std::left << "进度条"
              << std::endl;
    
    // 训练循环
    for (int epoch = 1; epoch <= config.epoch_num; ++epoch) {
        // 记录epoch开始时间
        auto epoch_start_time = std::chrono::steady_clock::now();
        
        // 训练阶段
        model->train();
        auto [train_loss, train_tokens, train_batches] = run_epoch(train_dataset, model, loss_compute_train,
                                                                  config.batch_size, device, config, true,
                                                                  epoch, config.epoch_num);
        
        // 验证阶段
        model->eval();
        auto [dev_loss, dev_tokens, dev_batches] = run_epoch(dev_dataset, model, loss_compute_eval,
                                                              config.batch_size, device, config, false,
                                                              epoch, config.epoch_num);
        
        // 计算BLEU分数（用于监控，但不用于保存模型）
        float bleu_score = evaluate(dev_dataset, model, config, device);
        
        // 计算epoch总时间
        auto epoch_end_time = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            epoch_end_time - epoch_start_time).count() / 1000.0;
        
        // 获取GPU内存
        std::string gpu_mem = "N/A";
        if (device.is_cuda()) {
            try {
                c10::cuda::CUDAGuard guard(device);
                size_t allocated = 0;
                size_t total = 0;
#ifdef USE_CUDA
                size_t free = 0;
                if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                    allocated = total - free;
                    double allocated_gb = allocated / (1024.0 * 1024.0 * 1024.0);
                    std::ostringstream gpu_oss;
                    gpu_oss << std::fixed << std::setprecision(1) << allocated_gb << "G";
                    gpu_mem = gpu_oss.str();
                }
#endif
            } catch (...) {
                gpu_mem = "N/A";
            }
        } else {
            gpu_mem = "0G";
        }
        
        // YOLOv5风格：表格格式输出epoch结果
        // 格式对齐表头：Epoch   GPU_mem   Batch   Tokens   train_loss   val_loss   BLEU     time
        // 示例：       1/100     2.5G   100/20     1.5M      0.1234     0.1456    12.34    45.6s
        
        // 格式化批次数量（显示训练和验证的批次，格式：train_batches/val_batches）
        std::ostringstream batch_oss;
        batch_oss << train_batches << "/" << dev_batches;
        
        // 计算每秒处理的token数量
        double tokens_per_sec = (epoch_duration > 0.0) ? (static_cast<double>(train_tokens) / epoch_duration) : 0.0;
        
        // 格式化每秒tokens数量（使用K/M/G等单位，添加/s后缀）
        std::string tokens_str;
        if (tokens_per_sec >= 1000000000) {
            std::ostringstream t_oss;
            t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000000000.0) << "G/s";
            tokens_str = t_oss.str();
        } else if (tokens_per_sec >= 1000000) {
            std::ostringstream t_oss;
            t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000000.0) << "M/s";
            tokens_str = t_oss.str();
        } else if (tokens_per_sec >= 1000) {
            std::ostringstream t_oss;
            t_oss << std::fixed << std::setprecision(1) << (tokens_per_sec / 1000.0) << "K/s";
            tokens_str = t_oss.str();
        } else {
            std::ostringstream t_oss;
            t_oss << std::fixed << std::setprecision(1) << tokens_per_sec << "/s";
            tokens_str = t_oss.str();
        }
        
        // YOLOv5风格：按照示例格式输出：val: 前缀，所有列左对齐，最后添加进度条（|====================| 100%）
        // 格式要与表头完全对齐
        std::string full_bar(20, '=');  // 100%进度条
        std::cout << "val: "
                  << std::setw(10) << std::left << (std::to_string(epoch) + "/" + std::to_string(config.epoch_num))
                  << std::setw(12) << std::left << gpu_mem
                  << std::setw(15) << std::left << batch_oss.str()
                  << std::setw(15) << std::left << tokens_str
                  << std::setw(15) << std::left << std::fixed << std::setprecision(4) << train_loss
                  << std::setw(15) << std::left << std::fixed << std::setprecision(4) << dev_loss
                  << std::setw(10) << std::left << std::fixed << std::setprecision(2) << bleu_score
                  << std::setw(10) << std::left << std::fixed << std::setprecision(1) << epoch_duration << "s"
                  << std::setw(28) << std::left << ("|" + full_bar + "| 100%")
                  << std::endl;
        
        // YOLOv5 风格：基于验证损失保存最佳模型
        // 如果当前验证损失小于历史最小损失，保存为 best.pth
        if (dev_loss < best_val_loss) {
            try {
                // 保存前清理CUDA缓存，释放未使用的显存
                if (device.is_cuda()) {
                 //   torch::cuda::empty_cache();
                }
                // 直接保存模型（不包含配置参数）
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
                // 保存后清理CUDA缓存
                if (device.is_cuda()) {
                  //  torch::cuda::empty_cache();
                }
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("保存最佳模型失败: ") + best_path + ", 错误: " + e.what());
            }
        }
        
        // YOLOv5 风格：每个 epoch 都保存 last.pth（覆盖之前的）
        try {
            // 保存前清理CUDA缓存
            if (device.is_cuda()) {
             //   torch::cuda::empty_cache();
            }
            // 直接保存模型（不包含配置参数）
            torch::save(model, last_path);
            {
                std::ostringstream oss;
                oss << "保存最后模型: " << last_path 
                    << " (Epoch " << epoch << ", ValLoss=" 
                    << std::fixed << std::setprecision(3) << dev_loss << ")";
                LOG_INFO(oss.str());
            }
            // 保存后清理CUDA缓存
            if (device.is_cuda()) {
            //   torch::cuda::empty_cache();
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
        
        // 显式释放 batch 中的张量（帮助释放显存）
        batch.src = torch::Tensor();
        batch.trg = torch::Tensor();
        batch.trg_y = torch::Tensor();
        batch.src_mask = torch::Tensor();
        batch.trg_mask = torch::Tensor();
        
        // 定期清理CUDA缓存（每10个batch清理一次）
        if (device.is_cuda() && (i + 1) % 10 == 0) {
            //torch::cuda::empty_cache();
        }
    }
    
    // 计算BLEU分数
    float bleu_score = corpus_bleu(all_candidates, all_references, 4);
    
    // 评估结束后清理CUDA缓存
    if (device.is_cuda()) {
      //  torch::cuda::empty_cache();
    }
    
    return bleu_score;
}

