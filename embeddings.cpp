#include "embeddings.h"
#include <cmath>

// Embeddings实现
EmbeddingsImpl::EmbeddingsImpl(int d_model, int vocab_size)
    : embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)),
      d_model(d_model) {
    register_module("embedding", embedding);
}

torch::Tensor EmbeddingsImpl::forward(torch::Tensor x) {
    // 返回embedding矩阵乘以sqrt(d_model)以保持方差
    return embedding->forward(x) * std::sqrt(static_cast<float>(d_model));
}

// PositionalEncoding实现
PositionalEncodingImpl::PositionalEncodingImpl(int d_model, float drop_rate, int max_len, torch::Device device)
    : dropout(torch::nn::DropoutOptions(drop_rate)) {
    register_module("dropout", dropout);
    
    // 生成位置索引 [max_len, 1]，直接在指定设备上创建
    auto position = torch::arange(0, max_len, torch::TensorOptions().dtype(torch::kFloat32).device(device)).unsqueeze(1);
    
    // 计算div_term: exp(-log(10000) / d_model * 2i)
    // 对于偶数维度: i = 0, 2, 4, ... -> div_term shape: [d_model/2]
    auto div_term = torch::exp(
        torch::arange(0, d_model, 2, torch::TensorOptions().dtype(torch::kFloat32).device(device)) * 
        (-std::log(10000.0f) / static_cast<float>(d_model))
    );
    
    // 计算位置编码: position * div_term shape: [max_len, d_model/2]
    auto pos_encoding = position * div_term;
    
    // 分别计算sin和cos值
    auto sin_vals = torch::sin(pos_encoding);  // [max_len, d_model/2]
    auto cos_vals = torch::cos(pos_encoding);  // [max_len, d_model/2]
    
    // 构建完整的pe矩阵：交替使用sin和cos值
    // 初始化pe为全零矩阵 [max_len, d_model]
    pe = torch::zeros({max_len, d_model}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // 使用index_put_来正确赋值
    // 为偶数维度（0, 2, 4, ...）赋值sin值
    // 为奇数维度（1, 3, 5, ...）赋值cos值
    int num_pairs = d_model / 2;
    for (int i = 0; i < num_pairs; ++i) {
        int even_idx = i * 2;
        int odd_idx = i * 2 + 1;
        
        // 使用index_put_为偶数维度赋值sin值
        pe.index_put_({torch::indexing::Slice(), even_idx}, sin_vals.select(1, i));
        
        // 使用index_put_为奇数维度赋值cos值（如果存在）
        if (odd_idx < d_model) {
            pe.index_put_({torch::indexing::Slice(), odd_idx}, cos_vals.select(1, i));
        }
    }
    
    // 如果d_model是奇数，最后一个维度使用sin
    if (d_model % 2 == 1 && num_pairs > 0) {
        pe.index_put_({torch::indexing::Slice(), d_model - 1}, sin_vals.select(1, num_pairs - 1));
    }
    
    // 增加batch维度: [1, max_len, d_model]
    pe = pe.unsqueeze(0);
    
    // 注册为buffer（不参与训练的参数），确保在正确的设备上
    register_buffer("pe", pe);
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) {
    // 将位置编码添加到输入
    // pe的形状应该是 [1, max_len, d_model]
    // x的形状是 [batch_size, seq_len, d_model]
    // Python版本: x = x + self.pe[:, :x.size(1)]
    // 这相当于取pe的 [:, 0:seq_len, :]，得到 [1, seq_len, d_model]
    // 然后通过广播机制与 [batch_size, seq_len, d_model] 相加
    
    // 获取序列长度
    int seq_len = x.size(1);
    
    // 取pe的前seq_len个位置: pe[:, 0:seq_len, :]
    // pe的形状: [1, max_len, d_model]
    // pe_slice的形状: [1, seq_len, d_model]
    auto pe_slice = pe.slice(1, 0, seq_len);
    
    // 确保pe_slice和x在同一个设备上
    if (pe_slice.device() != x.device()) {
        pe_slice = pe_slice.to(x.device());
    }
    
    // 广播相加: [batch_size, seq_len, d_model] + [1, seq_len, d_model]
    // LibTorch会自动处理广播
    x = x + pe_slice;
    return dropout->forward(x);
}

