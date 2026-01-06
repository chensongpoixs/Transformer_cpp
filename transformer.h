#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include <torch/torch.h>
#include "encoder.h"
#include "decoder.h"
#include "embeddings.h"
#include "generator.h"
#include <memory>

/**
 * Transformer模型
 * 完整的编码器-解码器架构
 */
class TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(
        Encoder encoder,
        Decoder decoder,
        Embeddings src_embed,
        Embeddings tgt_embed,
        Generator generator,
        PositionalEncoding src_position,
        PositionalEncoding tgt_position
    );
    
    torch::Tensor encode(torch::Tensor src, torch::Tensor src_mask);
    torch::Tensor decode(
        torch::Tensor memory,
        torch::Tensor src_mask,
        torch::Tensor tgt,
        torch::Tensor tgt_mask
    );
    torch::Tensor forward(
        torch::Tensor src,
        torch::Tensor tgt,
        torch::Tensor src_mask,
        torch::Tensor tgt_mask
    );

private:
    Encoder encoder;
    Decoder decoder;
    Embeddings src_embed;
    Embeddings tgt_embed;
    Generator generator;
    PositionalEncoding src_position;
    PositionalEncoding tgt_position;
};

TORCH_MODULE(Transformer);

/**
 * 构建完整的Transformer模型
 * @param src_vocab_size 源语言词汇表大小
 * @param tgt_vocab_size 目标语言词汇表大小
 * @param N 层数（默认6）
 * @param d_model 模型维度（默认512）
 * @param d_ff 前馈网络维度（默认2048）
 * @param h 多头注意力头数（默认8）
 * @param dropout Dropout率（默认0.1）
 * @param device 设备（默认CPU，训练时应使用CUDA）
 * @return Transformer模型
 */
Transformer make_model(
    int src_vocab_size,
    int tgt_vocab_size,
    int N = 6,
    int d_model = 512,
    int d_ff = 2048,
    int h = 8,
    float dropout = 0.1f,
    torch::Device device = torch::kCPU
);

#endif // TRANSFORMER_MODEL_H

