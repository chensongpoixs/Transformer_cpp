#ifndef TRANSFORMER_EMBEDDINGS_H
#define TRANSFORMER_EMBEDDINGS_H

#include <torch/torch.h>
#include <cmath>

/**
 * 词嵌入层
 * 将输入的离散词索引转换为连续的向量表示
 */
class EmbeddingsImpl : public torch::nn::Module {
public:
    EmbeddingsImpl(int d_model, int vocab_size);
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Embedding embedding;
    int d_model;
};

TORCH_MODULE(Embeddings);

/**
 * 位置编码层
 * 为每个输入位置添加唯一的位置编码
 */
class PositionalEncodingImpl : public torch::nn::Module {
public:
    PositionalEncodingImpl(int d_model, float dropout, int max_len = 5000, torch::Device device = torch::kCPU);
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Dropout dropout;
    torch::Tensor pe;  // 位置编码矩阵
};

TORCH_MODULE(PositionalEncoding);

#endif // TRANSFORMER_EMBEDDINGS_H

