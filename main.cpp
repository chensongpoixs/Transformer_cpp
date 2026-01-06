#include <iostream>
#include <torch/torch.h>
#include "transformer.h"
#include "config.h"
#include "train.h"
#include "train_utils.h"
#include "data_loader.h"
#include <memory>
#include <iomanip>

/**
 * Transformer训练主程序
 * 基于Python版本的C++实现
 */

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Transformer C++ Training Implementation" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载配置
    TransformerConfig config;
    
    // 设置设备
    torch::Device device(config.use_cuda && torch::cuda::is_available() 
                         ? torch::kCUDA 
                         : torch::kCPU);
    std::cout << "使用设备: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    if (device.is_cuda()) {
        std::cout << "GPU设备ID: " << config.device_id << std::endl;
    }
    
    try {
        // 加载数据集
        std::cout << "\n加载数据集..." << std::endl;
        MTDataset train_dataset(config.train_data_path);
        MTDataset dev_dataset(config.dev_data_path);
        std::cout << "训练集大小: " << train_dataset.size() << " 个样本" << std::endl;
        std::cout << "验证集大小: " << dev_dataset.size() << " 个样本" << std::endl;
        
        // 创建模型（直接在GPU上创建，所有buffer和参数都在GPU上）
        std::cout << "\n创建Transformer模型..." << std::endl;
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
        
        std::cout << "模型创建成功!" << std::endl;
        std::cout << "模型参数:" << std::endl;
        std::cout << "  - d_model: " << config.d_model << std::endl;
        std::cout << "  - n_heads: " << config.n_heads << std::endl;
        std::cout << "  - n_layers: " << config.n_layers << std::endl;
        std::cout << "  - d_ff: " << config.d_ff << std::endl;
        std::cout << "  - src_vocab_size: " << config.src_vocab_size << std::endl;
        std::cout << "  - tgt_vocab_size: " << config.tgt_vocab_size << std::endl;
        
        // 计算模型参数数量
        size_t num_params = 0;
        for (const auto& param : model->parameters()) {
            num_params += param.numel();
        }
        std::cout << "总参数量: " << num_params << std::endl;
        
        // 创建损失函数
        auto criterion = torch::nn::CrossEntropyLoss(
            torch::nn::CrossEntropyLossOptions()
                .ignore_index(config.padding_idx)
                .reduction(torch::kSum)
        );
        
        // 创建优化器
        std::cout << "\n创建优化器..." << std::endl;
        auto optimizer = get_std_opt(model, config.d_model);
        std::cout << "优化器创建成功 (NoamOpt with Adam)" << std::endl;
        
        // 开始训练
        std::cout << "\n开始训练..." << std::endl;
        std::cout << "训练配置:" << std::endl;
        std::cout << "  - 批次大小: " << config.batch_size << std::endl;
        std::cout << "  - 训练轮数: " << config.epoch_num << std::endl;
        std::cout << "  - 学习率: " << config.lr << std::endl;
        
        train(train_dataset, dev_dataset, model, criterion, optimizer, config, device);
        
        std::cout << "\n训练完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

