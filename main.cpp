#include <iostream>
#include <torch/torch.h>
#include "transformer.h"
#include "config.h"
#include <memory>

/**
 * Transformer训练主程序
 * 基于Python版本的C++实现
 */

int main(int argc, char* argv[]) {
    std::cout << "Transformer C++ Implementation" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // 加载配置
    TransformerConfig config;
    
    // 设置设备
    torch::Device device(config.use_cuda && torch::cuda::is_available() 
                         ? torch::kCUDA 
                         : torch::kCPU);
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    // 创建模型（直接在GPU上创建，所有buffer和参数都在GPU上）
    std::cout << "Creating Transformer model on " << (device.is_cuda() ? "GPU" : "CPU") << "..." << std::endl;
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
    
    model->train();
    
    std::cout << "Model created successfully!" << std::endl;
    std::cout << "Model parameters:" << std::endl;
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
    std::cout << "Total parameters: " << num_params << std::endl;
    
    // 示例：创建随机输入进行前向传播测试
    std::cout << "\nTesting forward pass..." << std::endl;
    int batch_size = 2;
    int src_len = 10;
    int tgt_len = 8;
    
    // Embedding需要Long类型的索引，必须显式指定dtype
    auto src = torch::randint(0, config.src_vocab_size, 
                              {batch_size, src_len}, 
                              torch::TensorOptions().dtype(torch::kLong).device(device));
    auto tgt = torch::randint(0, config.tgt_vocab_size, 
                              {batch_size, tgt_len}, 
                              torch::TensorOptions().dtype(torch::kLong).device(device));
    auto src_mask = (src != config.padding_idx).unsqueeze(1);
    auto tgt_mask = (tgt != config.padding_idx).unsqueeze(1);
    
    try {
        auto output = model->forward(src, tgt, src_mask, tgt_mask);
        std::cout << "Forward pass successful!" << std::endl;
        std::cout << "Output shape: [" << output.size(0) << ", " 
                  << output.size(1) << ", " << output.size(2) << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\nTransformer model is ready for training!" << std::endl;
    std::cout << "Note: Full training loop implementation requires data loading and optimization." << std::endl;
    
    return 0;
}

