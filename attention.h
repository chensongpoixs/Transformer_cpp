#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include <torch/torch.h>
#include <vector>
#include <cmath>

/**
 * 注意力机制实现
 */

/**
 * 计算注意力分数
 * @param query Query矩阵
 * @param key Key矩阵
 * @param value Value矩阵
 * @param mask 掩码（可选）
 * @param dropout Dropout层（可选）
 * @return 注意力输出和注意力权重
 */
std::pair<torch::Tensor, torch::Tensor> attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask = torch::Tensor(),
    torch::nn::Dropout dropout = torch::nn::Dropout(torch::nn::DropoutOptions(0.0f))
);

/**
 * 多头注意力机制
 */
class MultiHeadedAttentionImpl : public torch::nn::Module {
public:
    MultiHeadedAttentionImpl(int h, int d_model, float dropout = 0.1f);
    
    torch::Tensor forward(
        torch::Tensor query,
        torch::Tensor key,
        torch::Tensor value,
        torch::Tensor mask = torch::Tensor()
    );

private:
    int h;              // 头数
    int d_k;            // 每个头的维度
   // torch::nn::ModuleList linears;  // 4个线性层: WQ, WK, WV, WO
   torch::nn::Linear W_q;
    torch::nn::Linear W_k;
    torch::nn::Linear W_v;
    torch::nn::Linear output;
    torch::nn::Dropout dropout;
    torch::Tensor attn; // 注意力权重（用于可视化）
};

TORCH_MODULE(MultiHeadedAttention);

#endif // TRANSFORMER_ATTENTION_H

