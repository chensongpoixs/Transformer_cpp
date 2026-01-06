#ifndef TRANSFORMER_DECODER_H
#define TRANSFORMER_DECODER_H

#include <torch/torch.h>
#include "attention.h"
#include "feed_forward.h"
#include "sublayer_connection.h"
#include "layer_norm.h"

/**
 * 解码器层
 */
class DecoderLayerImpl : public torch::nn::Module {
public:
    DecoderLayerImpl(
        int size,
        MultiHeadedAttention self_attn,
        MultiHeadedAttention src_attn,
        PositionwiseFeedForward feed_forward,
        float dropout
    );
    
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor memory,
        torch::Tensor src_mask,
        torch::Tensor tgt_mask
    );
    int32_t laysize() { return size; }

private:
    MultiHeadedAttention self_attn;  // 自注意力
    MultiHeadedAttention src_attn;   // 与编码器输出的注意力
    PositionwiseFeedForward feed_forward;
    torch::nn::ModuleList sublayers;  // 3个子层连接
    int size;
};

TORCH_MODULE(DecoderLayer);

/**
 * 解码器
 * 由N个解码器层堆叠而成
 */
class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(DecoderLayer layer, int N);
    DecoderImpl(int d_model, int h, int d_ff, float dropout, int N);
    
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor memory,
        torch::Tensor src_mask,
        torch::Tensor tgt_mask
    );

private:
    torch::nn::ModuleList layers;
    //LayerNorm norm;
    torch::nn::LayerNorm norm;
    int d_model;
    int h;
    int d_ff;
    float dropout;
};

TORCH_MODULE(Decoder);

#endif // TRANSFORMER_DECODER_H

