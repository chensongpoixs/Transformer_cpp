#ifndef TRANSFORMER_CONFIG_H
#define TRANSFORMER_CONFIG_H

#include <string>

/**
 * Transformer模型配置类
 * 包含所有超参数和路径配置
 */
struct TransformerConfig {
    // 模型超参数
    int d_model = 512;              // 模型维度
    int n_heads = 8;                // 多头注意力头数
    int n_layers = 6;               // Transformer层数
    int d_k = 64;                   // 每个头的键向量维度
    int d_v = 64;                   // 每个头的值向量维度
    int d_ff = 2048;                // 前馈网络隐藏层维度
    float dropout = 0.1f;           // Dropout率
    
    // 词汇表配置
    int src_vocab_size = 32000;     // 源语言词汇表大小
    int tgt_vocab_size = 32000;     // 目标语言词汇表大小
    int padding_idx = 0;            // Padding标记索引
    int bos_idx = 2;                // 开始符索引
    int eos_idx = 3;                // 结束符索引
    
    // 训练配置
    int batch_size = 40;            // 批次大小
    int epoch_num = 100;            // 训练轮数
    float lr = 3e-4f;               // 学习率
    
    // 解码配置
    int max_len = 60;               // 最大序列长度
    int beam_size = 3;              // Beam Search大小
    
    // 文件路径
    std::string data_dir = "./data";
    std::string train_data_path = "./data/json/train.json";
    std::string dev_data_path = "./data/json/dev.json";
    std::string test_data_path = "./data/json/test.json";
    std::string model_path = "./weights/transformer_model.pth";
    
    // 设备配置
    bool use_cuda = true;           // 是否使用CUDA
    int device_id = 0;              // GPU设备ID
};

#endif // TRANSFORMER_CONFIG_H

