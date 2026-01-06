#include "attention.h"
#include <cmath>

std::pair<torch::Tensor, torch::Tensor> attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask,
    torch::nn::Dropout dropout) {
    
    int d_k = query.size(-1);
    
    // 计算注意力分数: Q * K^T / sqrt(d_k)
    auto scores = torch::matmul(query, key.transpose(-2, -1)) / std::sqrt(static_cast<float>(d_k));
    
    // 应用mask（如果提供）
    if (mask.defined() && mask.numel() > 0) {
        scores = scores.masked_fill(mask == 0, -1e9);
    }
    
    // Softmax归一化
    auto p_attn = torch::softmax(scores, -1);
    
    // 应用dropout
    if (!dropout.is_empty() ) {
        p_attn = dropout(p_attn);
    }
    
    // 返回注意力输出和注意力权重
    return {torch::matmul(p_attn, value), p_attn};
}

// MultiHeadedAttention实现
MultiHeadedAttentionImpl::MultiHeadedAttentionImpl(int h, int d_model, float drop_rate)
    : h(h), d_k(d_model / h), 
    W_q(torch::nn::LinearOptions(d_model, d_model) ),
    W_k(torch::nn::LinearOptions(d_model, d_model) ),
    W_v(torch::nn::LinearOptions(d_model, d_model) ),
    output(torch::nn::LinearOptions(d_model, d_model)),
    dropout(torch::nn::DropoutOptions(drop_rate)) {
    //printf("====<>>>>d_model = %u\n", d_model);
    // 确保d_model可以被h整除
    assert(d_model % h == 0);
    
    // 创建4个线性层: WQ, WK, WV, WO
   /* for (int i = 0; i < 4; ++i) {
        linears->push_back(torch::nn::Linear(d_model, d_model));
    }*/
    //W_q = std::move(torch::nn::LinearOptions(d_model, d_model));// torch::nn::Linear(d_model, d_model);
    //W_k = torch::nn::Linear(d_model, d_model);
    //W_v = torch::nn::Linear(d_model, d_model);
    //output = torch::nn::Linear(d_model, d_model);
    register_module("W_q", W_q);
    register_module("W_k", W_k);
    register_module("W_v", W_v);
    register_module("output", output);
    //register_module("linears", linears);
    register_module("dropout", dropout);
}

torch::Tensor MultiHeadedAttentionImpl::forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask) {
    
    int nbatches = query.size(0);
    
    // 如果mask存在，增加维度
    if (mask.defined() && mask.numel() > 0) {
        mask = mask.unsqueeze(1);
    }
    //return mask;
    // 1) 通过线性层并重塑为多头形式
    // [batch, seq_len, d_model] -> [batch, h, seq_len, d_k]
   // auto q = linears->at<torch::nn::Linear>(0)->forward(query).view({nbatches, -1, h, d_k}).transpose(1, 2);
    //auto k = linears->at<torch::nn::Linear>(1)->forward(key).view({nbatches, -1, h, d_k}).transpose(1, 2);
    //auto v = linears->at<torch::nn::Linear>(2)->forward(value).view({nbatches, -1, h, d_k}).transpose(1, 2);
    auto q = W_q->forward(query).view({nbatches, -1, h, d_k}).transpose(1, 2);
    auto k = W_k->forward(key).view({nbatches, -1, h, d_k}).transpose(1, 2);
    auto v = W_v->forward(value).view({nbatches, -1, h, d_k}).transpose(1, 2);
    // 2) 计算注意力
    auto result = attention(q, k, v, mask, dropout);
    auto x = result.first;
    attn = result.second;
    
    // 3) 拼接多头: [batch, h, seq_len, d_k] -> [batch, seq_len, d_model]
    x = x.transpose(1, 2).contiguous().view({nbatches, -1, h * d_k});
    
    // 4) 通过输出线性层
    
   // return mask;
    //return linears->at<torch::nn::Linear>(3)->forward(x);
    return output->forward(x);
}

