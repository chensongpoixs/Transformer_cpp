/******************************************************************************
 *  Copyright (c) 2026 The Transformer project authors . All Rights Reserved.
 *
 *  Please visit https://chensongpoixs.github.io for detail
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 ******************************************************************************/
/*****************************************************************************
				   Author: chensong
				   date:  2026-01-01
 * 解码器实现 (Decoder Implementation)
 * 
 * 实现解码器层和完整解码器的前向传播
 * 
 * DecoderLayer: 单个解码器层，包含掩码自注意力、交叉注意力和前馈网络
 * Decoder: 由多个解码器层堆叠而成的完整解码器
				   
				   
				   
				   
 输赢不重要，答案对你们有什么意义才重要。

 光阴者，百代之过客也，唯有奋力奔跑，方能生风起时，是时代造英雄，英雄存在于时代。或许世人道你轻狂，可你本就年少啊。 看护好，自己的理想和激情。


 我可能会遇到很多的人，听他们讲好2多的故事，我来写成故事或编成歌，用我学来的各种乐器演奏它。
 然后还可能在一个国家遇到一个心仪我的姑娘，她可能会被我帅气的外表捕获，又会被我深邃的内涵吸引，在某个下雨的夜晚，她会全身淋透然后要在我狭小的住处换身上的湿衣服。
 3小时候后她告诉我她其实是这个国家的公主，她愿意向父皇求婚。我不得已告诉她我是穿越而来的男主角，我始终要回到自己的世界。
 然后我的身影慢慢消失，我看到她眼里的泪水，心里却没有任何痛苦，我才知道，原来我的心被丢掉了，我游历全世界的原因，就是要找回自己的本心。
 于是我开始有意寻找各种各样失去心的人，我变成一块砖头，一颗树，一滴水，一朵白云，去听大家为什么会失去自己的本心。
 我发现，刚出生的宝宝，本心还在，慢慢的，他们的本心就会消失，收到了各种黑暗之光的侵蚀。
 从一次争论，到嫉妒和悲愤，还有委屈和痛苦，我看到一只只无形的手，把他们的本心扯碎，蒙蔽，偷走，再也回不到主人都身边。
 我叫他本心猎手。他可能是和宇宙同在的级别 但是我并不害怕，我仔细回忆自己平淡的一生 寻找本心猎手的痕迹。
 沿着自己的回忆，一个个的场景忽闪而过，最后发现，我的本心，在我写代码的时候，会回来。
 安静，淡然，代码就是我的一切，写代码就是我本心回归的最好方式，我还没找到本心猎手，但我相信，顺着这个线索，我一定能顺藤摸瓜，把他揪出来。

 ******************************************************************************/

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

