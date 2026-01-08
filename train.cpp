/******************************************************************************
 *  Copyright (c) 2026 The Transformer project authors . All Rights Reserved.
 *
 *  Please visit https://chensongpoixs.github.io for detail
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 ******************************************************************************/
/*****************************************************************************
				   Author: chensong
				   date:  2026-01-01
 * 训练实现 (Training Implementation)
 * 
 * 实现完整的训练流程，包括：
 * - run_epoch: 运行一个 epoch 的训练或验证，支持 bucket 采样
 * - train: 主训练函数，包含训练循环、验证、模型保存等
 * - evaluate: 评估函数，使用 beam search 解码并计算 BLEU 分数
 * - save_config_file: 保存训练配置到 config.yaml（YOLOv5 风格）
 * 
 * 训练特性：
 * - YOLOv5 风格的进度显示和日志输出
 * - 基于验证损失保存最佳模型
 * - 支持 bucket 采样提高训练效率
				   
				   
				   
				   
 输赢不重要，答案对你们有什么意义才重要。

 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。


 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。

 ******************************************************************************/

#include "train.h"
#include "beam_search.h"
#include "multi_process_loader.h"
#include "data_cache.h"
#include "amp_scaler.h"
#include "bleu.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include "gpu_profiler.h"
#include "cuda_stream_manager.h"
#include "resource_manager.h"
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
#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ctime>
#include <future>
#include <thread>


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
        LOG_WARN("Failed to create project directory: " + project_path.string() + ", error: " + ec.message());
    }

    // 首先尝试 project/name
    fs::path exp_dir = project_path / name;
    if (!fs::exists(exp_dir) || exist_ok) {
        if (exist_ok && fs::exists(exp_dir)) {
            LOG_INFO("Experiment directory already exists, use existing directory: " + exp_dir.string());
        }
        fs::create_directories(exp_dir / "weights", ec);
        if (ec) {
            LOG_WARN("Failed to create weights directory: " + (exp_dir / "weights").string() + ", error: " + ec.message());
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
                LOG_WARN("Failed to create weights directory: " + (exp_dir_i / "weights").string() + ", error: " + ec.message());
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
        LOG_WARN("Failed to save training config file: " + config_path);
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
    LOG_INFO("Training config saved to: " + config_path);
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

// 辅助函数：根据当前 batch 索引获取一个 Batch（单线程模式）
static Batch get_batch_for_index(size_t i,
                                 int batch_size,
                                 const std::vector<size_t>& indices,
                                 MTDataset& dataset,
                                 torch::Device device,
                                 const TransformerConfig& config,
                                 std::unique_ptr<CudaStreamManager>& stream_manager,
                                 double& collate_time_ms) {
    size_t start_idx = i * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, indices.size());
    std::vector<size_t> batch_indices(indices.begin() + start_idx,
                                     indices.begin() + end_idx);
    
    auto collate_start = steady_clock::now();
    GPUProfiler::start_timer("collate_fn");
    
    if (device.is_cuda() && stream_manager && i > 0) {
        // 在传输 Stream 上准备当前 batch 的数据
        stream_manager->set_current_stream(0);
    }
    
    Batch b = dataset.collate_fn(batch_indices, device,
                                 config.padding_idx, config.bos_idx, config.eos_idx,
                                 config.src_vocab_size, config.tgt_vocab_size);
    
    GPUProfiler::end_timer("collate_fn");
    auto collate_end = steady_clock::now();
    collate_time_ms = duration_cast<microseconds>(collate_end - collate_start).count() / 1000.0;
    return b;
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
    // CUDA Stream 管理器：用于可选的流水线并行（根据配置决定是否启用）
    std::unique_ptr<CudaStreamManager> stream_manager;
    if (device.is_cuda() && config.use_cuda_stream) {
        // ✅ 阶段 2：N 个 Stream 实现深度流水线（可配置）
        // Stream 0: 数据传输（Batch N+1）
        // Stream 1: 前向传播（Batch N）
        // Stream 2: 反向传播（Batch N）
        // Stream 3+: 额外的数据传输或计算流（如果 stream_count >= 4）
        int stream_count = std::max(2, std::min(config.cuda_stream_count, 8));  // 限制在 2-8 之间
        stream_manager = std::make_unique<CudaStreamManager>(device, stream_count);
        LOG_INFO("Using " + std::to_string(stream_count) + " CUDA Streams for deep pipeline parallelism");
    } else if (device.is_cuda() && !config.use_cuda_stream) {
        LOG_INFO("CUDA Stream disabled, using default CUDA stream");
    }

    float total_tokens = 0.0f;
    float total_loss = 0.0f;
    
    // ✅ 方案 1：延迟 loss 提取 - 累积 loss tensor，批量提取
    std::vector<torch::Tensor> loss_tensor_buffer;  // 累积 loss tensor
    std::vector<float> ntokens_buffer;              // 对应的 token 数量
    const size_t LOSS_EXTRACT_INTERVAL = 10;        // 每 10 个 batch 提取一次
    
    // ✅ 方案 2 + 方案 3：Event 同步 + 减少同步频率（业界标准 + YOLOv5 策略）
    at::cuda::CUDAEvent compute_event;
    const size_t SYNC_INTERVAL = 10;  // 每 10 个 batch 同步一次（与延迟 loss 提取一致）
    bool event_initialized = false;
    
    // ✅ 阶段 2：4 个 Stream 深度流水线 - Event 管理
    at::cuda::CUDAEvent transfer_event;      // 数据传输完成事件
    at::cuda::CUDAEvent forward_event;       // 前向传播完成事件
    at::cuda::CUDAEvent backward_event;      // 反向传播完成事件
    bool events_initialized = false;
    
    // 基于句子长度的 bucket 采样策略
    // 1. 先按长度排序得到索引
    LOG_DEBUG("Start bucket sampling: dataset size = " + std::to_string(dataset.size()));
    auto bucket_start_time = steady_clock::now();
    std::vector<size_t> base_indices = dataset.make_length_sorted_indices();
    auto bucket_end_time = steady_clock::now();
    double bucket_time = duration_cast<milliseconds>(bucket_end_time - bucket_start_time).count() / 1000.0;
    LOG_DEBUG("Length sorting finished: num_indices=" + std::to_string(base_indices.size()) + ", time=" + std::to_string(bucket_time) + "s");

    // 2. 按 bucket 切分，再在 bucket 内部打乱
    std::vector<size_t> indices;
    indices.reserve(base_indices.size());

    const size_t bucket_size = static_cast<size_t>(batch_size) * 4;  // 可调：4 倍batch
    std::vector<size_t> bucket;
    bucket.reserve(bucket_size);

    std::random_device rd;
    std::mt19937 g(rd());

    size_t bucket_count = 0;
    size_t total_buckets = (base_indices.size() + bucket_size - 1) / bucket_size;
    LOG_DEBUG("Bucket config: bucket_size=" + std::to_string(bucket_size) + ", estimated_num_buckets=" + std::to_string(total_buckets));

    // 记录初始显存
    size_t mem_before_bucket = 0;
    if (device.is_cuda()) {
        try {
            auto stats = GPUProfiler::get_memory_stats(device);
            mem_before_bucket = stats.allocated_bytes_current;
            LOG_DEBUG("Memory before bucket sampling: " + std::to_string(mem_before_bucket / 1024 / 1024) + "MB");
        } catch (...) {
            LOG_WARN("Failed to get initial GPU memory info");
        }
    }

    for (size_t idx : base_indices) {
        bucket.push_back(idx);
        if (bucket.size() >= bucket_size) {
            // 打乱 bucket 内部的顺序
            std::shuffle(bucket.begin(), bucket.end(), g);
            indices.insert(indices.end(), bucket.begin(), bucket.end());
            bucket_count++;
            
            // 记录每个 bucket 处理后的显存（每10个bucket记录一次）
            if (device.is_cuda() && bucket_count % 10 == 0) {
                try {
                    auto stats = GPUProfiler::get_memory_stats(device);
                    size_t mem_current = stats.allocated_bytes_current;
                    size_t mem_diff = mem_current - mem_before_bucket;
                    LOG_DEBUG("Bucket " + std::to_string(bucket_count) + "/" + std::to_string(total_buckets) + 
                             ": allocated=" + std::to_string(mem_current / 1024 / 1024) + "MB, " +
                             "increase=" + std::to_string(mem_diff / 1024 / 1024) + "MB");
                } catch (...) {
                    LOG_WARN("Exception occurred while getting bucket memory stats (ignored)");
                }
            }
            
            bucket.clear();
        }
    }
    // 处理最后一个不满的 bucket
    if (!bucket.empty()) {
        std::shuffle(bucket.begin(), bucket.end(), g);
        indices.insert(indices.end(), bucket.begin(), bucket.end());
        bucket_count++;
    }
    
    LOG_DEBUG("Bucket sampling finished: num_buckets=" + std::to_string(bucket_count) + 
             ", num_indices=" + std::to_string(indices.size()));
    
    // 记录 bucket 采样后的显存
    if (device.is_cuda()) {
        try {
            auto stats = GPUProfiler::get_memory_stats(device);
            size_t mem_after_bucket = stats.allocated_bytes_current;
            size_t mem_diff = mem_after_bucket - mem_before_bucket;
            LOG_DEBUG("Memory after bucket sampling: " + std::to_string(mem_after_bucket / 1024 / 1024) + "MB, " +
                     "increase=" + std::to_string(mem_diff / 1024 / 1024) + "MB");
        } catch (...) {
            LOG_WARN("Failed to get memory stats after bucket sampling");
        }
    }
    
    // 按批次处理数据
    size_t num_batches = (indices.size() + batch_size - 1) / batch_size;
    LOG_DEBUG("Start batch processing: num_batches=" + std::to_string(num_batches) + ", batch_size=" + std::to_string(batch_size));
    
    // 计时相关
    auto epoch_start = steady_clock::now();
    size_t processed_samples = 0;
    
    // 记录批次处理前的显存
    size_t mem_before_batches = 0;
    if (device.is_cuda()) {
        try {
            auto stats = GPUProfiler::get_memory_stats(device);
            mem_before_batches = stats.allocated_bytes_current;
            LOG_DEBUG("Memory before batch processing: " + std::to_string(mem_before_batches / 1024 / 1024) + "MB");
        } catch (...) {
            LOG_WARN("Failed to get memory stats before batch processing");
        }
    }
    
    // ✅ 阶段 3：数据缓存（如果启用）- 使用 RAII 管理
    std::unique_ptr<DataCache> data_cache;
    bool use_data_cache = (config.cache_size > 0 && device.is_cuda());
    DataCacheRAII data_cache_guard(nullptr);  // RAII 包装，确保 stop() 被调用
    
    // ✅ 使用多进程数据加载器（如果 workers > 0）
    std::unique_ptr<MultiProcessDataLoader> multi_loader;
    bool use_multi_loader = (config.workers > 0);
    
    if (use_data_cache) {
        // 创建数据缓存（预加载多个 batch 到 GPU）
        data_cache = std::make_unique<DataCache>(config.cache_size, device);
        data_cache->start_prefetch(dataset, indices, batch_size, config);
        // 使用 RAII 包装，确保在作用域结束时自动调用 stop()
        data_cache_guard = DataCacheRAII(data_cache.get());
        LOG_INFO("Using GPU data cache: cache_size=" + std::to_string(config.cache_size));
    } 
    if (use_multi_loader) {
        // 创建多进程数据加载器
        multi_loader = std::make_unique<MultiProcessDataLoader>(
            dataset, indices, batch_size, device, config,
            config.workers, config.pin_memory, config.prefetch_factor
        );
        LOG_INFO("Using multi-process data loader: workers=" + std::to_string(config.workers) +
                 ", pin_memory=" + std::string(config.pin_memory ? "true" : "false"));
    }
    
    // ✅ 阶段 3：混合精度训练（如果启用）
    std::unique_ptr<AMPScaler> amp_scaler;
    bool use_amp = (config.use_amp && device.is_cuda() && is_training);
    if (use_amp) {
        amp_scaler = std::make_unique<AMPScaler>(config.amp_init_scale, config.amp_scale_window);
        LOG_INFO("Using mixed precision training (FP16): init_scale=" + 
                 std::to_string(config.amp_init_scale) + ", scale_window=" + 
                 std::to_string(config.amp_scale_window));
    }
    
    for (size_t i = 0; i < num_batches; ++i) {
        double collate_time_ms = 0.0;
        
        // ✅ 阶段 3：优先使用数据缓存，其次多进程加载器，最后单线程加载
        // 使用 RAII 确保 Batch 中的张量在作用域结束时自动释放
        Batch batch;
        
        if (use_data_cache && data_cache) {
            auto collate_start = steady_clock::now();
            batch = data_cache->get_next();
            auto collate_end = steady_clock::now();
            collate_time_ms = duration_cast<microseconds>(collate_end - collate_start).count() / 1000.0;
            
            // 检查是否加载完成
            if (!batch.src.defined()) {
                LOG_DEBUG("Data cache finished at batch " + std::to_string(i));
                break;
            }
        } else if (use_multi_loader && multi_loader) {
            auto collate_start = steady_clock::now();
            batch = multi_loader->next();
            auto collate_end = steady_clock::now();
            collate_time_ms = duration_cast<microseconds>(collate_end - collate_start).count() / 1000.0;
            
            // 检查是否加载完成
            if (!batch.src.defined()) {
                LOG_DEBUG("Data loader finished at batch " + std::to_string(i));
                break;
            }
        } else {
            // 单线程模式：使用原有逻辑
            batch = get_batch_for_index(i, batch_size, indices, dataset, device,
                                        config, stream_manager, collate_time_ms);
        }
        
        // 在 batch 赋值完成后创建 RAII guard，确保在循环结束时自动释放
        BatchScopeGuard batch_guard(batch);  // RAII 保护，确保张量释放
        
        // ✅ 阶段 2：4 个 Stream 深度流水线 + Event 同步（业界标准）
        if (device.is_cuda() && stream_manager) {
            // 初始化所有 Event（第一个 batch）
            if (!events_initialized) {
                transfer_event = stream_manager->create_event();
                forward_event = stream_manager->create_event();
                backward_event = stream_manager->create_event();
                compute_event = stream_manager->create_event();
                events_initialized = true;
                event_initialized = true;
            }
            
            if (i == 0) {
                // 第一个 batch：等待数据传输完成
                stream_manager->synchronize(0);  // 第一个 batch 需要同步传输
                // 记录传输完成事件
                stream_manager->set_current_stream(0);
                stream_manager->record_event(transfer_event, 0);
            } else {
                // 后续 batch：使用 Event 同步 Stream 依赖
                // 只在必要时同步（每 10 个 batch 或最后一个 batch）
                bool should_sync = ((i + 1) % SYNC_INTERVAL == 0) || (i == num_batches - 1);
                
                if (should_sync) {
                    // 批量同步：等待上一个 batch 的计算完成
                    backward_event.synchronize();
                } else {
                    // 非阻塞检查：不阻塞 CPU
                    if (!stream_manager->query_event(backward_event)) {
                        // 事件未完成，但不等待，让 GPU 继续工作
                    }
                }
                
                // Stream 0: 记录当前 batch 的传输完成事件
                // 数据传输已在 collate_fn 中完成（使用 non_blocking=true）
                stream_manager->set_current_stream(0);
                stream_manager->record_event(transfer_event, 0);
            }
            
            // Stream 1: 前向传播（等待传输完成）
            if (stream_manager->num_streams() >= 2) {
                // ✅ 修复：使用 CudaStreamManager 的 wait_event_on_stream 方法
                stream_manager->wait_event_on_stream(transfer_event, 1);  // Stream 1 等待传输完成
            }
            stream_manager->set_current_stream(1);
        }
        
        // ✅ 阶段 3：前向传播（支持混合精度训练）
        // 验证阶段使用 NoGradGuard 避免构建计算图，节省显存
        torch::Tensor out;
        auto forward_start = steady_clock::now();
        if (is_training) {
            GPUProfiler::start_timer("forward");
            if (use_amp) {
                // 混合精度训练：将输入转换为 FP16
                auto src_fp16 = batch.src.to(torch::kFloat16);
                auto trg_fp16 = batch.trg.to(torch::kFloat16);
                auto src_mask_fp16 = batch.src_mask.to(torch::kFloat16);
                auto trg_mask_fp16 = batch.trg_mask.to(torch::kFloat16);
                
                // 前向传播（FP16）
                out = model->forward(src_fp16, trg_fp16, src_mask_fp16, trg_mask_fp16);
                // 输出转换为 FP32（用于 loss 计算）
                out = out.to(torch::kFloat32);
            } else {
                // FP32 训练
                out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
            }
            GPUProfiler::end_timer("forward");
        } else {
            torch::NoGradGuard no_grad;
            GPUProfiler::start_timer("forward");
            out = model->forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask);
            GPUProfiler::end_timer("forward");
        }
        auto forward_end = steady_clock::now();
        double forward_time_ms = duration_cast<microseconds>(forward_end - forward_start).count() / 1000.0;
        
        // ✅ 阶段 2：记录前向传播完成事件
        if (device.is_cuda() && stream_manager && events_initialized) {
            stream_manager->record_event(forward_event, 1);  // 在 Stream 1 上记录前向完成事件
        }
        
        // ✅ 阶段 2：Stream 2 等待前向传播完成
        if (device.is_cuda() && stream_manager && stream_manager->num_streams() >= 3) {
            // ✅ 修复：使用 CudaStreamManager 的 wait_event_on_stream 方法
            stream_manager->wait_event_on_stream(forward_event, 2);  // Stream 2 等待前向完成
            stream_manager->set_current_stream(2);  // 切换到 Stream 2 进行反向传播
        }
        
        // ✅ 阶段 3：计算损失（支持混合精度训练）
        auto loss_start = steady_clock::now();
        GPUProfiler::start_timer("loss_compute");
        
        torch::Tensor loss_tensor;
        bool has_backward = false;
        
        if (use_amp && amp_scaler && is_training) {
            // 混合精度训练：分离反向传播和优化器更新
            // 1. 计算 loss（不执行反向传播）
            loss_tensor = loss_compute.compute_loss_and_backward(
                out, batch.trg_y, static_cast<float>(batch.ntokens));
            
            // 2. 缩放 loss（在反向传播之前）
            loss_tensor = amp_scaler->scale(loss_tensor);
            
            // 3. 执行反向传播（loss 已缩放）
            loss_tensor.backward();
            
            // 4. 取消缩放梯度
            auto base_optimizer = loss_compute.get_base_optimizer();
            if (base_optimizer) {
                amp_scaler->unscale(base_optimizer);
            }
            
            // 5. 如果梯度溢出，跳过优化器更新
            if (!amp_scaler->has_overflow()) {
                loss_compute.optimizer_step();
                has_backward = true;
            } else {
                // 梯度溢出，跳过更新
                LOG_WARN("Gradient overflow detected, skipping optimizer step");
            }
            
            // 6. 更新缩放因子
            amp_scaler->update();
        } else {
            // 标准训练：使用原有方法
            std::tie(loss_tensor, has_backward) = loss_compute.compute_loss_tensor(
                out, batch.trg_y, static_cast<float>(batch.ntokens));
        }
        
        GPUProfiler::end_timer("loss_compute");
        auto loss_end = steady_clock::now();
        double loss_time_ms = duration_cast<microseconds>(loss_end - loss_start).count() / 1000.0;
        
        // ✅ 阶段 2：记录反向传播完成事件（在 Stream 2 上）
        if (device.is_cuda() && stream_manager && events_initialized) {
            if (stream_manager->num_streams() >= 3) {
                stream_manager->record_event(backward_event, 2);  // 在 Stream 2 上记录反向完成事件
            } else {
                // 如果只有 2 个 Stream，在 Stream 1 上记录
                stream_manager->record_event(backward_event, 1);
            }
            // 同时记录 compute_event（用于兼容原有逻辑）
            stream_manager->record_event(compute_event, (stream_manager->num_streams() >= 3) ? 2 : 1);
        }
        
        // ✅ 延迟提取：累积 loss tensor，批量提取
        loss_tensor_buffer.push_back(loss_tensor);
        ntokens_buffer.push_back(static_cast<float>(batch.ntokens));
        
        // 累加 token 数量（立即累加，用于统计）
        total_tokens += batch.ntokens;
        
        // 每 N 个 batch 或最后一个 batch 时，批量提取 loss 值
        bool should_extract = ((i + 1) % LOSS_EXTRACT_INTERVAL == 0) || (i == num_batches - 1);
        
        float current_loss = 0.0f;  // 当前 batch 的 loss（用于显示）
        if (should_extract && !loss_tensor_buffer.empty()) {
            // 批量提取所有累积的 loss 值（减少同步次数）
            for (size_t j = 0; j < loss_tensor_buffer.size(); ++j) {
                float loss_value = loss_tensor_buffer[j].item<float>();  // 批量同步
                total_loss += loss_value * ntokens_buffer[j];
                
                // 最后一个 loss 用于当前显示
                if (j == loss_tensor_buffer.size() - 1) {
                    current_loss = loss_value;
                }
                
                // 释放 loss tensor
                loss_tensor_buffer[j] = torch::Tensor();
            }
            loss_tensor_buffer.clear();
            ntokens_buffer.clear();
        } else {
            // 如果不需要提取，使用估算值（基于历史平均值）
            // 注意：这只是用于显示，实际累加会在批量提取时进行
            float avg_loss_so_far = (total_tokens > 0.0f) ? (total_loss / total_tokens) : 0.0f;
            current_loss = avg_loss_so_far;  // 使用平均值作为临时显示值
        }
        
        size_t current_batch_size = static_cast<size_t>(batch.src.size(0));
        processed_samples += current_batch_size;
        
        // ✅ 立即释放所有张量（关键修复：防止显存泄漏）
        // 使用 RAII：batch_guard 会在作用域结束时自动释放 Batch 中的张量
        // 但为了及时释放，我们显式释放 out 张量
        out = torch::Tensor();
        // Batch 中的张量由 batch_guard 在作用域结束时自动释放
        // 如果需要立即释放，可以调用 batch_guard.release()
        
        // 注意：data_cache_guard 会在函数返回时自动调用 stop()，无需手动调用
    
    // ✅ 优化：减少显存统计频率，避免频繁同步
    // 每 50 个 batch 或每个 bucket 结束时记录（减少同步操作）
    if (device.is_cuda() && ((i + 1) % 50 == 0 || (i + 1) % bucket_size == 0)) {
            try {
                auto stats = GPUProfiler::get_memory_stats(device);
                size_t mem_current = stats.allocated_bytes_current;
                size_t mem_reserved = stats.reserved_bytes_current;
                size_t mem_diff = mem_current - mem_before_batches;
                
                // 判断是否在 bucket 边界
                bool is_bucket_end = ((i + 1) % bucket_size == 0);
                std::string log_prefix = is_bucket_end ? "[Bucket end] " : "";
                
                LOG_DEBUG(log_prefix + "Batch " + std::to_string(i + 1) + "/" + std::to_string(num_batches) +
                         ": allocated=" + std::to_string(mem_current / 1024 / 1024) + "MB, " +
                         "reserved=" + std::to_string(mem_reserved / 1024 / 1024) + "MB, " +
                         "increase=" + std::to_string(mem_diff / 1024 / 1024) + "MB");
                
                // 如果是 bucket 结束，强制清理 CUDA 缓存
                if (is_bucket_end) {
                    // Python: torch.cuda.empty_cache()
                    // C++: 使用 c10::cuda::CUDACachingAllocator::emptyCache() 清理缓存
                    c10::cuda::CUDACachingAllocator::emptyCache();
                    auto stats_after = GPUProfiler::get_memory_stats(device);
                    size_t mem_after_cache = stats_after.allocated_bytes_current;
                    LOG_DEBUG("[Bucket end] Memory after empty cache: " + std::to_string(mem_after_cache / 1024 / 1024) + "MB, " +
                             "released=" + std::to_string((mem_current - mem_after_cache) / 1024 / 1024) + "MB");
                }
            } catch (...) {
                LOG_WARN("Exception occurred while getting batch memory stats or emptying cache (ignored)");
            }
        }
        
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
        
        // ✅ 优化：减少进度条更新频率，避免频繁输出影响性能
        // 每 10 个 batch 或最后一个 batch 更新一次
        if (i % 10 == 0 || i == num_batches - 1) {
            print_progress_bar(epoch, total_epochs, i, num_batches,
                              current_loss, avg_loss_so_far, speed, eta, is_training, device, elapsed_time,
                              static_cast<long long>(total_tokens), num_batches);
        }
        
        // 定期清理CUDA缓存（每50个batch清理一次，避免频繁清理影响性能）
        if (device.is_cuda() && (i + 1) % 50 == 0) {
           // torch::cuda::empty_cache();  // ✅ 启用：强制释放 CUDA 缓存
        }
    }
    
    // ✅ 确保所有累积的 loss tensor 都已提取（防止遗漏）
    if (!loss_tensor_buffer.empty()) {
        // ✅ 阶段 2：确保最后一个 batch 的所有操作完成（批量同步）
        if (device.is_cuda() && stream_manager && events_initialized) {
            // 同步所有 Stream 上的操作
            backward_event.synchronize();  // 确保反向传播完成
            compute_event.synchronize();   // 确保所有计算完成
        }
        
        for (size_t j = 0; j < loss_tensor_buffer.size(); ++j) {
            float loss_value = loss_tensor_buffer[j].item<float>();
            total_loss += loss_value * ntokens_buffer[j];
            loss_tensor_buffer[j] = torch::Tensor();
        }
        loss_tensor_buffer.clear();
        ntokens_buffer.clear();
    }
    
    
    // ✅ 性能瓶颈诊断：在第一个epoch结束后打印详细分析
    if (epoch == 1 && is_training) {
        GPUProfiler::print_summary();
        GPUProfiler::check_gpu_utilization(device);
        
        // 计算平均时间（从 GPUProfiler 获取）
        auto collate_info = GPUProfiler::get_timing_info("collate_fn");
        auto forward_info = GPUProfiler::get_timing_info("forward");
        auto loss_info = GPUProfiler::get_timing_info("loss_compute");
        
        double avg_collate = (collate_info.count > 0) ? (collate_info.total_time_ms / collate_info.count) : 0.0;
        double avg_forward = (forward_info.count > 0) ? (forward_info.total_time_ms / forward_info.count) : 0.0;
        double avg_loss = (loss_info.count > 0) ? (loss_info.total_time_ms / loss_info.count) : 0.0;
        int collate_count = collate_info.count;
        int forward_count = forward_info.count;
        int loss_count = loss_info.count;
        
        // 估算总 batch 时间（假设其他时间为 10%）
        double estimated_total = (avg_collate + avg_forward + avg_loss) / 0.9;
        double collate_ratio = (estimated_total > 0) ? (avg_collate / estimated_total * 100.0) : 0.0;
        double compute_ratio = (estimated_total > 0) ? ((avg_forward + avg_loss) / estimated_total * 100.0) : 0.0;
        double other_ratio = 100.0 - collate_ratio - compute_ratio;
        
        // ✅ 详细性能瓶颈诊断
        LOG_INFO("========== Performance Bottleneck Diagnosis ==========");
        LOG_INFO("Time Distribution (from GPUProfiler):");
        LOG_INFO("  Data loading (collate_fn): " + std::to_string(collate_ratio) + "% (" + 
                 std::to_string(avg_collate) + " ms, " + std::to_string(collate_count) + " calls)");
        LOG_INFO("  GPU computation (forward+loss): " + std::to_string(compute_ratio) + "% (" + 
                 std::to_string(avg_forward + avg_loss) + " ms)");
        LOG_INFO("  Other (sync/wait/overhead): " + std::to_string(other_ratio) + "%");
        LOG_INFO("");
        
        // 识别瓶颈并给出建议
        bool has_bottleneck = false;
        
        if (collate_ratio > 50.0) {
            has_bottleneck = true;
            LOG_WARN("🔴 BOTTLENECK: Data loading is the bottleneck!");
            LOG_INFO("  Current configuration:");
            LOG_INFO("    --workers: " + std::to_string(config.workers) + 
                     (config.workers == 0 ? " (single-threaded)" : " (multi-threaded)"));
            LOG_INFO("    --pin-memory: " + std::string(config.pin_memory ? "true" : "false"));
            LOG_INFO("    --prefetch-factor: " + std::to_string(config.prefetch_factor));
            LOG_INFO("    --cache-size: " + std::to_string(config.cache_size));
            LOG_INFO("  Recommendations:");
            if (config.workers == 0) {
                LOG_INFO("    1. ⭐ Enable multi-process loading: --workers 8");
            }
            if (config.cache_size == 0) {
                LOG_INFO("    2. ⭐ Enable GPU data cache: --cache-size 2");
            }
            if (!config.pin_memory) {
                LOG_INFO("    3. ⭐ Enable pin_memory: --pin-memory true");
            }
            if (config.prefetch_factor < 2) {
                LOG_INFO("    4. ⭐ Increase prefetch: --prefetch-factor 4");
            }
            LOG_INFO("");
        }
        
        if (compute_ratio < 30.0) {
            has_bottleneck = true;
            LOG_WARN("🔴 BOTTLENECK: GPU computation time is too low!");
            LOG_INFO("  Current configuration:");
            LOG_INFO("    --batch-size: " + std::to_string(config.batch_size));
            LOG_INFO("    --d-model: " + std::to_string(config.d_model));
            LOG_INFO("    --n-layers: " + std::to_string(config.n_layers));
            LOG_INFO("    --use-cuda-stream: " + std::string(config.use_cuda_stream ? "true" : "false"));
            LOG_INFO("  Recommendations:");
            if (config.batch_size < 64) {
                LOG_INFO("    1. ⭐ Increase batch size: --batch-size 64 (or 128)");
            }
            if (!config.use_cuda_stream) {
                LOG_INFO("    2. ⭐ Enable CUDA Stream: --use-cuda-stream true");
            }
            if (config.d_model < 512 || config.n_layers < 6) {
                LOG_INFO("    3. Consider increasing model size: --d-model 512 --n-layers 6");
            }
            LOG_INFO("");
        }
        
        if (other_ratio > 20.0) {
            has_bottleneck = true;
            LOG_WARN("🟠 WARNING: High synchronization/wait time!");
            LOG_INFO("  Recommendations:");
            LOG_INFO("    1. ⭐ Enable CUDA Stream: --use-cuda-stream true");
            LOG_INFO("    2. Loss extraction is already optimized (every 10 batches)");
            LOG_INFO("    3. Memory stats frequency is already optimized (every 50 batches)");
            LOG_INFO("");
        }
        
        // GPU 利用率估算
        double estimated_gpu_util = compute_ratio;
        if (estimated_gpu_util < 30.0) {
            LOG_WARN("🔴 GPU utilization is very low: " + std::to_string(estimated_gpu_util) + "%");
        } else if (estimated_gpu_util < 60.0) {
            LOG_WARN("🟠 GPU utilization is moderate: " + std::to_string(estimated_gpu_util) + "%");
        } else {
            LOG_INFO("✅ GPU utilization is good: " + std::to_string(estimated_gpu_util) + "%");
        }
        
        if (!has_bottleneck) {
            LOG_INFO("✅ No major bottlenecks detected. Performance looks good!");
        }
        
        LOG_INFO("=====================================================");
        LOG_INFO("For detailed analysis, see: PERFORMANCE_BOTTLENECK_ANALYSIS.md");
        LOG_INFO("=====================================================");
    }
    
    // epoch结束后清理CUDA缓存（使用 CUDACachingAllocator::emptyCache 替代 torch::cuda::empty_cache）
    if (device.is_cuda()) {
        try {
            auto stats_before = GPUProfiler::get_memory_stats(device);
            size_t mem_before = stats_before.allocated_bytes_current;
            
            // Python: torch.cuda.empty_cache()
            // C++: 使用 c10::cuda::CUDACachingAllocator::emptyCache() 清理缓存
            c10::cuda::CUDACachingAllocator::emptyCache();
            torch::cuda::synchronize();  // 确保所有 CUDA 内存释放后同步

            auto stats_after = GPUProfiler::get_memory_stats(device);
            size_t mem_after = stats_after.allocated_bytes_current;
            
            LOG_DEBUG("Clear cache at epoch end: before=" + std::to_string(mem_before / 1024 / 1024) + "MB, " +
                     "after=" + std::to_string(mem_after / 1024 / 1024) + "MB, " +
                     "released=" + std::to_string((mem_before - mem_after) / 1024.0 / 1024.0) + "MB");
        } catch (...) {
            LOG_WARN("Failed to get memory stats at epoch end");
            c10::cuda::CUDACachingAllocator::emptyCache();
            torch::cuda::synchronize();
        }
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
    LOG_INFO("Project dir: " + config.project);
    LOG_INFO("Experiment name: " + config.name);
    LOG_INFO("Experiment dir: " + exp_folder);
    LOG_INFO("Weights dir: " + weights_folder);
    
    // 设置日志文件路径（默认写入到实验目录）
    std::string log_file_path = exp_folder + "/training.log";
    Logger::set_log_file(log_file_path);
    LOG_INFO("Log file: " + log_file_path);
    
    // 保存训练配置文件（YOLOv5 风格）
    save_config_file(config, exp_folder);
    
    // YOLOv5 风格：基于验证损失保存最佳模型
    float best_val_loss = std::numeric_limits<float>::infinity();  // 最小验证损失
    std::string best_path = weights_folder + "/best.pth";
    std::string last_path = weights_folder + "/last.pth";
    
    // 创建损失计算器
    auto loss_compute_train = LossCompute(model->get_generator(), criterion, optimizer);
    auto loss_compute_eval = LossCompute(model->get_generator(), criterion, nullptr);
    LOG_INFO("LossCompute objects created (train & eval)");
    
    // 计算训练数据集的bucket采样信息（在训练开始前打印）
    const size_t bucket_size = static_cast<size_t>(config.batch_size) * 4;  // 可调：4 倍batch
    size_t train_dataset_size = train_dataset.size();
    size_t train_num_batches = (train_dataset_size + config.batch_size - 1) / config.batch_size;
    LOG_INFO("Using length-based bucket sampling: bucket_size=" + std::to_string(bucket_size) +
             ", num_samples=" + std::to_string(train_dataset_size) + ", num_batches=" + std::to_string(train_num_batches));
    
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
           // torch::cuda::empty_cache();  // ✅ 启用：强制释放 CUDA 缓存
        }
    }
    
    // 计算BLEU分数
    float bleu_score = corpus_bleu(all_candidates, all_references, 4);
    
    // 评估结束后清理CUDA缓存
    if (device.is_cuda()) {
       // torch::cuda::empty_cache();  // ✅ 启用：强制释放 CUDA 缓存
    }
    
    return bleu_score;
}


