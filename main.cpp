#include <iostream>
#include <torch/torch.h>
#include "transformer.h"
#include "config.h"
#include "train.h"
#include "train_utils.h"
#include "data_loader.h"
#include "logger.h"
#include <memory>
#include <iomanip>

using namespace logging;

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
    
    // 加载配置
    TransformerConfig config;
    
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
            << ", train_data_path=" << config.train_data_path
            << ", dev_data_path=" << config.dev_data_path;
        LOG_INFO(oss.str());
    }
    
    try {
        // 加载数据集
        LOG_INFO("开始加载数据集...");
        MTDataset train_dataset(config.train_data_path);
        MTDataset dev_dataset(config.dev_data_path);
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

