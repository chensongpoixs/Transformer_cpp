#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include <torch/torch.h>
#include "attention.h"
#include "feed_forward.h"
#include "sublayer_connection.h"
#include "layer_norm.h"
#include <vector>

/**
 * 编码器层
 */
class EncoderLayerImpl : public torch::nn::Module {
public:
    EncoderLayerImpl(
        int size,
        MultiHeadedAttention self_attn,
        PositionwiseFeedForward feed_forward,
        float dropout
    );
    
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);

    int32_t laysize() { return size; }
private:
    MultiHeadedAttention self_attn;
    PositionwiseFeedForward feed_forward;
    torch::nn::ModuleList sublayers;
    int size;
};

TORCH_MODULE(EncoderLayer);

/**
 * 编码器
 * 由N个编码器层堆叠而成
 */
class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(EncoderLayer layer, int N);
    EncoderImpl(int d_model, int h, int d_ff, float dropout, int N);
    
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);

private:
    torch::nn::ModuleList layers;
    LayerNorm norm;
   // torch::nn::LayerNorm norm;
    int d_model;
    int h;
    int d_ff;
    float dropout;
};

TORCH_MODULE(Encoder);

#endif // TRANSFORMER_ENCODER_H

