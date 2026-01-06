#include "encoder.h"
#include "utils.h"

// EncoderLayer实现
EncoderLayerImpl::EncoderLayerImpl(
    int size,
    MultiHeadedAttention self_attn,
    PositionwiseFeedForward feed_forward,
    float dropout)
    : self_attn(self_attn), feed_forward(feed_forward), size(size) {
    
    register_module("self_attn", self_attn);
    register_module("feed_forward", feed_forward);
    
    // 创建2个子层连接: self-attention和feed-forward
    sublayers->push_back(SublayerConnection(size, dropout));
    sublayers->push_back(SublayerConnection(size, dropout));
    
    register_module("sublayers", sublayers);
}

torch::Tensor EncoderLayerImpl::forward(torch::Tensor x, torch::Tensor mask) {
    // 自注意力 + 残差连接 + Layer Norm
    auto self_attn_mask = mask;
    x = sublayers[0]->as<SublayerConnection>()->forward(
        x, 
        [this, self_attn_mask](torch::Tensor x) -> torch::Tensor {
            return self_attn->forward(x, x, x, self_attn_mask);
        }
    );
    
    // 前馈网络 + 残差连接 + Layer Norm
    return sublayers[1]->as<SublayerConnection>()->forward(x, [this](torch::Tensor x) -> torch::Tensor {
        return feed_forward->forward(x);
    });
}

// Encoder实现（使用layer模板）
EncoderImpl::EncoderImpl(EncoderLayer layer, int N)
    : norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ layer->laysize() }))),
    //norm(LayerNorm(layer->laysize())),
      d_model(layer->laysize()),
      h(8),  // 默认值，应该从配置获取
      d_ff(2048),  // 默认值，应该从配置获取
      dropout(0.1f) {  // 默认值，应该从配置获取
    
    // 创建N个独立的编码器层
    for (int i = 0; i < N; ++i) {
        // 为每一层创建新的注意力模块和前馈网络
        auto self_attn = MultiHeadedAttention(h, d_model, dropout);
        auto ff = PositionwiseFeedForward(d_model, d_ff, dropout);
        layers->push_back(EncoderLayer(d_model, self_attn, ff, dropout));
    }
    
    register_module("layers", layers);
    register_module("norm", norm);
}

// Encoder实现（直接使用参数）
EncoderImpl::EncoderImpl(int d_model, int h, int d_ff, float dropout, int N)
    :norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d_model }))),
    //: norm(LayerNorm(d_model)),
      d_model(d_model),
      h(h),
      d_ff(d_ff),
      dropout(dropout) {
    
    // 创建N个独立的编码器层
    for (int i = 0; i < N; ++i) {
        auto self_attn = MultiHeadedAttention(h, d_model, dropout);
        auto ff = PositionwiseFeedForward(d_model, d_ff, dropout);
        layers->push_back(EncoderLayer(d_model, self_attn, ff, dropout));
    }
    
    register_module("layers", layers);
    register_module("norm", norm);
}

torch::Tensor EncoderImpl::forward(torch::Tensor x, torch::Tensor mask) {
    // 通过所有编码器层
    for (auto& layer : *layers) {
        x = layer->as<EncoderLayer>()->forward(x, mask);
    }
    // 最终层归一化
    return norm->forward(x);
}

