#include "decoder.h"
#include "utils.h"

// DecoderLayer实现
DecoderLayerImpl::DecoderLayerImpl(
    int size,
    MultiHeadedAttention self_attn,
    MultiHeadedAttention src_attn,
    PositionwiseFeedForward feed_forward,
    float dropout)
    : self_attn(self_attn), src_attn(src_attn), feed_forward(feed_forward), size(size) {
    
    register_module("self_attn", self_attn);
    register_module("src_attn", src_attn);
    register_module("feed_forward", feed_forward);
    
    // 创建3个子层连接: self-attention, src-attention, feed-forward
    sublayers->push_back(SublayerConnection(size, dropout));
    sublayers->push_back(SublayerConnection(size, dropout));
    sublayers->push_back(SublayerConnection(size, dropout));
    
    register_module("sublayers", sublayers);
}

torch::Tensor DecoderLayerImpl::forward(
    torch::Tensor x,
    torch::Tensor memory,
    torch::Tensor src_mask,
    torch::Tensor tgt_mask) {
    
    // 1. 自注意力 + 残差连接 + Layer Norm
    auto tgt_mask_copy = tgt_mask;
    x = sublayers[0]->as<SublayerConnection>()->forward(
        x,
        [this, tgt_mask_copy](torch::Tensor x) -> torch::Tensor {
            return self_attn->forward(x, x, x, tgt_mask_copy);
        }
    );
    
    // 2. 与编码器输出的注意力 + 残差连接 + Layer Norm
    auto memory_copy = memory;
    auto src_mask_copy = src_mask;
    x = sublayers[1]->as<SublayerConnection>()->forward(
        x,
        [this, memory_copy, src_mask_copy](torch::Tensor x) -> torch::Tensor {
            return src_attn->forward(x, memory_copy, memory_copy, src_mask_copy);
        }
    );
    
    // 3. 前馈网络 + 残差连接 + Layer Norm
    return sublayers[2]->as<SublayerConnection>()->forward(x, [this](torch::Tensor x) -> torch::Tensor {
        return feed_forward->forward(x);
    });
}

// Decoder实现（使用layer模板）
DecoderImpl::DecoderImpl(DecoderLayer layer, int N)
    //: norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ layer->laysize() }))),
      :norm(LayerNorm(layer->laysize())),
      d_model(layer->laysize()),
      h(8),  // 默认值，应该从配置获取
      d_ff(2048),  // 默认值，应该从配置获取
      dropout(0.1f) {  // 默认值，应该从配置获取
    
    // 创建N个独立的解码器层
    for (int i = 0; i < N; ++i) {
        auto self_attn = MultiHeadedAttention(h, d_model, dropout);
        auto src_attn = MultiHeadedAttention(h, d_model, dropout);
        auto ff = PositionwiseFeedForward(d_model, d_ff, dropout);
        layers->push_back(DecoderLayer(d_model, self_attn, src_attn, ff, dropout));
    }
    
    register_module("layers", layers);
    register_module("norm", norm);
}

// Decoder实现（直接使用参数）
DecoderImpl::DecoderImpl(int d_model, int h, int d_ff, float dropout, int N)
   // : norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }))),
    : norm(LayerNorm(d_model)),
      d_model(d_model),
      h(h),
      d_ff(d_ff),
      dropout(dropout) {
    
    // 创建N个独立的解码器层
    for (int i = 0; i < N; ++i) {
        auto self_attn = MultiHeadedAttention(h, d_model, dropout);
        auto src_attn = MultiHeadedAttention(h, d_model, dropout);
        auto ff = PositionwiseFeedForward(d_model, d_ff, dropout);
        layers->push_back(DecoderLayer(d_model, self_attn, src_attn, ff, dropout));
    }
    
    register_module("layers", layers);
    register_module("norm", norm);
}

torch::Tensor DecoderImpl::forward(
    torch::Tensor x,
    torch::Tensor memory,
    torch::Tensor src_mask,
    torch::Tensor tgt_mask) {
    
    // 通过所有解码器层
    for (auto& layer : *layers) {
        x = layer->as<DecoderLayer>()->forward(x, memory, src_mask, tgt_mask);
    }
    // 最终层归一化
    return norm->forward(x);
}

