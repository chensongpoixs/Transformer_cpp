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
 * 注意力机制实现 (Attention Mechanism Implementation)
 * 
 * 实现缩放点积注意力和多头注意力机制
 * 
 * 核心计算：
 * - attention: 计算注意力分数和输出
 * - MultiHeadedAttention: 多头注意力的前向传播
				   
				   
				   
				   
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
    std::pair<torch::Tensor, torch::Tensor> result = attention(q, k, v, mask, dropout);
    auto x = result.first;
    attn = result.second;
    
    // 3) 拼接多头: [batch, h, seq_len, d_k] -> [batch, seq_len, d_model]
    x = x.transpose(1, 2).contiguous().view({nbatches, -1, h * d_k});
    
    // 4) 通过输出线性层
    
   // return mask;
    //return linears->at<torch::nn::Linear>(3)->forward(x);
    return output->forward(x);
}

