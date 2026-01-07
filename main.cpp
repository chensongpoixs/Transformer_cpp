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
 * 主程序入口 (Main Entry Point)
 * 
 * 这是 Transformer 训练程序的主入口文件，负责：
 * - 解析命令行参数（YOLOv5 风格）
 * - 初始化配置和数据集
 * - 创建模型和优化器
 * - 启动训练或测试流程
 * 
 * 命令行参数：
 * --data: 数据目录
 * --batch-size: 批次大小
 * --epochs: 训练轮数
 * --lr: 学习率
 * --weights: 预训练权重路径
 * --project: 项目目录
 * --name: 实验名称
				   
				   
				   
				   
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

#include <iostream>
#include <torch/torch.h>
#include "transformer.h"
#include "config.h"
#include "train.h"
#include "train_utils.h"
#include "data_loader.h"
#include "tokenizer_wrapper.h"
#include "logger.h"
#include <memory>
#include <iomanip>
#include <filesystem>

using namespace logging;
namespace fs = std::filesystem;

/**
 * YOLOv5 风格命令行解析
 * 参考 YOLOv5 train.py 的参数风格
 */
static bool parse_args(int argc, char* argv[], TransformerConfig& config) {
    std::string data_dir_arg;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](int& idx) -> const char* {
            if (idx + 1 >= argc) {
                LOG_WARN("命令行参数缺少值: " + arg);
                return nullptr;
            }
            return argv[++idx];
        };

        // YOLOv5 风格参数（使用短横线）
        if (arg == "--data") {
            if (auto v = next(i)) data_dir_arg = v;
        } else if (arg == "--batch-size") {
            if (auto v = next(i)) config.batch_size = std::stoi(v);
        } else if (arg == "--epochs") {
            if (auto v = next(i)) config.epoch_num = std::stoi(v);
        } else if (arg == "--lr") {
            if (auto v = next(i)) config.lr = std::stof(v);
        } else if (arg == "--d-model") {
            if (auto v = next(i)) config.d_model = std::stoi(v);
        } else if (arg == "--n-layers") {
            if (auto v = next(i)) config.n_layers = std::stoi(v);
        } else if (arg == "--n-heads") {
            if (auto v = next(i)) config.n_heads = std::stoi(v);
        } else if (arg == "--d-ff") {
            if (auto v = next(i)) config.d_ff = std::stoi(v);
        } else if (arg == "--dropout") {
            if (auto v = next(i)) config.dropout = std::stof(v);
        } else if (arg == "--beam-size") {
            if (auto v = next(i)) config.beam_size = std::stoi(v);
        } else if (arg == "--device") {
            if (auto v = next(i)) {
                std::string device_str = v;
                if (device_str == "cpu") {
                    config.use_cuda = false;
                } else {
                    try {
                        config.device_id = std::stoi(device_str);
                        config.use_cuda = true;
                    } catch (...) {
                        LOG_WARN("无效的设备ID: " + device_str + "，使用默认值");
                    }
                }
            }
        } else if (arg == "--project") {
            if (auto v = next(i)) config.project = v;
        } else if (arg == "--name") {
            if (auto v = next(i)) config.name = v;
        } else if (arg == "--weights") {
            if (auto v = next(i)) config.weights = v;
        } else if (arg == "--resume") {
            if (auto v = next(i)) config.resume = v;
        } else if (arg == "--workers") {
            if (auto v = next(i)) config.workers = std::stoi(v);
        } else if (arg == "--exist-ok") {
            config.exist_ok = true;
        } else if (arg == "--tokenizer-dir") {
            if (auto v = next(i)) {
                config.tokenizer_dir = v;
                // 如果只指定了目录，自动拼接默认文件名
                fs::path base(v);
                config.tokenizer_eng = (base / "eng.model").string();
                config.tokenizer_chn = (base / "chn.model").string();
            }
        } else if (arg == "--tokenizer-eng") {
            if (auto v = next(i)) config.tokenizer_eng = v;
        } else if (arg == "--tokenizer-chn") {
            if (auto v = next(i)) config.tokenizer_chn = v;
        } else if (arg == "--prefetch") {
            if (auto v = next(i)) {
                std::string mode = v;
                if (mode == "none") {
                    config.prefetch_mode = 0;
                } else if (mode == "async") {
                    config.prefetch_mode = 1;
                } else if (mode == "thread") {
                    config.prefetch_mode = 2;
                } else {
                    try {
                        int m = std::stoi(mode);
                        if (m >= 0 && m <= 2) {
                            config.prefetch_mode = m;
                        } else {
                            LOG_WARN("无效的 --prefetch 值(需为 0/1/2 或 none/async/thread): " + mode +
                                     "，使用默认值 " + std::to_string(config.prefetch_mode));
                        }
                    } catch (...) {
                        LOG_WARN("无法解析 --prefetch 参数: " + mode +
                                 "，使用默认值 " + std::to_string(config.prefetch_mode));
                    }
                }
            }
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else {
            LOG_WARN("未知命令行参数: " + arg);
        }
    }

    if (show_help) {
        LOG_INFO("Transformer C++ 训练程序 - YOLOv5 风格命令行参数");
        LOG_INFO("");
        LOG_INFO("数据相关:");
        LOG_INFO("  --data <path>              数据目录（包含 train.json/dev.json/test.json）");
        LOG_INFO("");
        LOG_INFO("训练相关:");
        LOG_INFO("  --batch-size <int>         批次大小 (默认: " + std::to_string(config.batch_size) + ")");
        LOG_INFO("  --epochs <int>             训练轮数 (默认: " + std::to_string(config.epoch_num) + ")");
        LOG_INFO("  --lr <float>               学习率 (默认: " + std::to_string(config.lr) + ")");
        LOG_INFO("  --workers <int>            数据加载线程数 (默认: " + std::to_string(config.workers) + ", 0=单线程)");
        LOG_INFO("  --prefetch <mode>          数据预取模式: none|async|thread (默认: " +
                 std::string(config.prefetch_mode == 0 ? "none" :
                             config.prefetch_mode == 1 ? "async" : "thread") + ")");
        LOG_INFO("");
        LOG_INFO("模型相关:");
        LOG_INFO("  --d-model <int>            模型维度 (默认: " + std::to_string(config.d_model) + ")");
        LOG_INFO("  --n-layers <int>           Transformer层数 (默认: " + std::to_string(config.n_layers) + ")");
        LOG_INFO("  --n-heads <int>            多头注意力头数 (默认: " + std::to_string(config.n_heads) + ")");
        LOG_INFO("  --d-ff <int>               前馈网络隐藏层维度 (默认: " + std::to_string(config.d_ff) + ")");
        LOG_INFO("  --dropout <float>          Dropout率 (默认: " + std::to_string(config.dropout) + ")");
        LOG_INFO("  --beam-size <int>          Beam Search大小 (默认: " + std::to_string(config.beam_size) + ")");
        LOG_INFO("");
        LOG_INFO("实验相关:");
        LOG_INFO("  --project <path>           项目目录 (默认: " + config.project + ")");
        LOG_INFO("  --name <str>               实验名称 (默认: " + config.name + ")");
        LOG_INFO("  --exist-ok                 如果实验目录已存在则覆盖");
        LOG_INFO("");
        LOG_INFO("分词器相关:");
        LOG_INFO("  --tokenizer-dir <path>      分词器目录 (默认: " + config.tokenizer_dir + ")");
        LOG_INFO("  --tokenizer-eng <path>      英文分词器模型路径 (默认: " + config.tokenizer_eng + ")");
        LOG_INFO("  --tokenizer-chn <path>      中文分词器模型路径 (默认: " + config.tokenizer_chn + ")");
        LOG_INFO("");
        LOG_INFO("其他:");
        LOG_INFO("  --device <int|cpu>         设备 (默认: " + std::to_string(config.device_id) + ", 或 'cpu')");
        LOG_INFO("  --weights <path>           预训练权重路径");
        LOG_INFO("  --resume <path>            恢复训练的检查点路径");
        LOG_INFO("  --help, -h                 显示此帮助信息");
        LOG_INFO("");
        LOG_INFO("示例:");
        LOG_INFO("  transformer.exe --data D:/data/mt --batch-size 64 --epochs 50");
        LOG_INFO("  transformer.exe --data ./data --project runs/train --name exp1 --exist-ok");
        return false;  // 返回 false 表示应该退出程序
    }

    // 处理 data_dir：如果命令行传入，则覆盖默认值，并自动拼接 json 路径
    if (!data_dir_arg.empty()) {
        fs::path base(data_dir_arg);
        config.data_dir = base.string();

        fs::path train_p = base / "train.json";
        fs::path dev_p   = base / "dev.json";
        fs::path test_p  = base / "test.json";

        config.train_data_path = train_p.string();
        config.dev_data_path   = dev_p.string();
        config.test_data_path  = test_p.string();

        LOG_INFO("使用命令行指定的数据目录: " + config.data_dir);
        LOG_INFO("  train_data_path = " + config.train_data_path);
        LOG_INFO("  dev_data_path   = " + config.dev_data_path);
        LOG_INFO("  test_data_path  = " + config.test_data_path);
    } else {
        // 未指定 data_dir，则沿用 config 默认值
        LOG_INFO("未通过命令行指定 --data，使用默认路径:");
        LOG_INFO("  train_data_path = " + config.train_data_path);
        LOG_INFO("  dev_data_path   = " + config.dev_data_path);
        LOG_INFO("  test_data_path  = " + config.test_data_path);
    }
    
    // 输出分词器路径信息
    LOG_INFO("分词器路径:");
    LOG_INFO("  tokenizer_eng = " + config.tokenizer_eng);
    LOG_INFO("  tokenizer_chn = " + config.tokenizer_chn);
    
    return true;  // 返回 true 表示继续执行程序
}

/**
 * Transformer训练主程序
 * 基于Python版本的C++实现
 */

int main(int argc, char* argv[]) {
    // 初始化日志系统
    Logger::init(Level::Debug);
    LOG_INFO("========================================");
    LOG_INFO("Transformer C++ Training Implementation");
    LOG_INFO("========================================");
    
    // 加载配置（先用默认值，再用命令行参数覆盖）
    TransformerConfig config;
    if (!parse_args(argc, argv, config)) {
        // 如果 parse_args 返回 false（例如用户请求 --help），则退出
        // 修复日志中bug 
        std::this_thread::sleep_for(std::chrono::seconds(3));
       // printf("==\n");
        return 0;
    }
    
    // 设置设备
    torch::Device device(config.use_cuda && torch::cuda::is_available() 
                         ? torch::kCUDA 
                         : torch::kCPU);
    LOG_INFO(std::string("使用设备: ") + (device.is_cuda() ? "CUDA" : "CPU"));
    if (device.is_cuda()) {
        LOG_INFO("GPU设备ID: " + std::to_string(config.device_id));
    }

    {
        std::ostringstream oss;
        oss << "配置: batch_size=" << config.batch_size
            << ", epoch_num=" << config.epoch_num
            << ", lr=" << config.lr
            << ", prefetch_mode=" << config.prefetch_mode
            << ", train_data_path=" << config.train_data_path
            << ", dev_data_path=" << config.dev_data_path;
        LOG_INFO(oss.str());
    }
    
    try {
        // 加载数据集
        LOG_INFO("开始加载数据集...");
        MTDataset train_dataset(config.train_data_path);
        MTDataset dev_dataset(config.dev_data_path);
        
        // 使用配置的分词器路径加载分词器
        LOG_INFO("加载分词器...");
        auto eng_tokenizer = english_tokenizer_load(config.tokenizer_eng);
        auto chn_tokenizer = chinese_tokenizer_load(config.tokenizer_chn);
        
        // 为数据集设置分词器
        train_dataset.set_tokenizers(eng_tokenizer, chn_tokenizer);
        dev_dataset.set_tokenizers(eng_tokenizer, chn_tokenizer);
        
        {
            std::ostringstream oss;
            oss << "数据集加载完成: 训练集样本数=" << train_dataset.size()
                << ", 验证集样本数=" << dev_dataset.size();
            LOG_INFO(oss.str());
        }
        
        // 创建模型（直接在GPU上创建，所有buffer和参数都在GPU上）
        LOG_INFO("开始创建 Transformer 模型...");
        auto model = make_model(
            config.src_vocab_size,
            config.tgt_vocab_size,
            config.n_layers,
            config.d_model,
            config.d_ff,
            config.n_heads,
            config.dropout,
            device  // 直接在指定设备上创建模型
        );
        
        LOG_INFO("模型创建成功!");
        {
            std::ostringstream oss;
            oss << "模型参数: "
                << "d_model=" << config.d_model
                << ", n_heads=" << config.n_heads
                << ", n_layers=" << config.n_layers
                << ", d_ff=" << config.d_ff
                << ", src_vocab_size=" << config.src_vocab_size
                << ", tgt_vocab_size=" << config.tgt_vocab_size;
            LOG_INFO(oss.str());
        }
        
        // 计算模型参数数量
        size_t num_params = 0;
        for (const auto& param : model->parameters()) {
            num_params += param.numel();
        }
        LOG_INFO("总参数量: " + std::to_string(num_params));
        
        // 加载预训练权重（如果指定了 --weights）
        if (!config.weights.empty()) {
            try {
                if (fs::exists(config.weights)) {
                    torch::load(model, config.weights, device);
                    LOG_INFO("成功加载预训练权重: " + config.weights);
                } else {
                    LOG_WARN("权重文件不存在: " + config.weights);
                }
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("加载权重失败: ") + config.weights + ", 错误: " + e.what());
            }
        }
        
        // 恢复训练（如果指定了 --resume）
        int start_epoch = 1;
        if (!config.resume.empty()) {
            try {
                if (fs::exists(config.resume)) {
                    torch::load(model, config.resume, device);
                    // 注意：这里简化处理，实际应该也加载优化器状态和 epoch 编号
                    // 为了完整实现，需要保存/加载更多状态信息
                    LOG_INFO("成功恢复训练检查点: " + config.resume);
                    LOG_WARN("注意：当前实现仅恢复模型权重，未恢复优化器状态和 epoch 编号");
                } else {
                    LOG_WARN("检查点文件不存在: " + config.resume);
                }
            } catch (const std::exception& e) {
                LOG_ERROR(std::string("恢复训练失败: ") + config.resume + ", 错误: " + e.what());
            }
        }
        
        // 创建损失函数
        auto criterion = torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions()
                .ignore_index(config.padding_idx)
                .reduction(torch::kSum)
        );
        LOG_INFO("CrossEntropyLoss 损失函数创建完成");
        
        // 创建优化器
        LOG_INFO("创建优化器 (NoamOpt + Adam)...");
        auto optimizer = get_std_opt(model, config.d_model);
        LOG_INFO("优化器创建成功");
        
        // 开始训练
        {
            std::ostringstream oss;
            oss << "开始训练: batch_size=" << config.batch_size
                << ", epoch_num=" << config.epoch_num
                << ", lr=" << config.lr;
            LOG_INFO(oss.str());
        }
        
        train(train_dataset, dev_dataset, model, criterion, optimizer, config, device);
        
        LOG_INFO("训练完成");
        
    } catch (const std::exception& e) {
        LOG_ERROR(std::string("程序异常: ") + e.what());
        return 1;
    }
    
    return 0;
}

