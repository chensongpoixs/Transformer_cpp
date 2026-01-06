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
 * Transformer 模型实现 (Transformer Model Implementation)
 * 
 * 实现完整的 Transformer 编码器-解码器架构
 * 
 * 主要功能：
 * - encode: 编码源语言序列
 * - decode: 解码生成目标语言序列
 * - forward: 完整的前向传播流程
 * - make_model: 创建并初始化 Transformer 模型
				   
				   
				   
				   
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

#include "transformer.h"
#include "utils.h"
#include <torch/nn/init.h>

// Transformer实现
TransformerImpl::TransformerImpl(
    Encoder encoder,
    Decoder decoder,
    Embeddings src_embed,
    Embeddings tgt_embed,
    Generator generator,
    PositionalEncoding src_position,
    PositionalEncoding tgt_position)
    : encoder(encoder),
      decoder(decoder),
      src_embed(src_embed),
      tgt_embed(tgt_embed),
      generator(generator),
      src_position(src_position),
      tgt_position(tgt_position) {
    
    register_module("encoder", encoder);
    register_module("decoder", decoder);
    register_module("src_embed", src_embed);
    register_module("tgt_embed", tgt_embed);
    register_module("generator", generator);
    register_module("src_position", src_position);
    register_module("tgt_position", tgt_position);
}

torch::Tensor TransformerImpl::encode(torch::Tensor src, torch::Tensor src_mask) {
    auto x = src_embed->forward(src);
    x = src_position->forward(x);
    return encoder->forward(x, src_mask);
}

torch::Tensor TransformerImpl::decode(
    torch::Tensor memory,
    torch::Tensor src_mask,
    torch::Tensor tgt,
    torch::Tensor tgt_mask) {
    auto x = tgt_embed->forward(tgt);
    x = tgt_position->forward(x);
    return decoder->forward(x, memory, src_mask, tgt_mask);
}

torch::Tensor TransformerImpl::forward(
    torch::Tensor src,
    torch::Tensor tgt,
    torch::Tensor src_mask,
    torch::Tensor tgt_mask) {
    // encoder的结果作为decoder的memory参数传入
    return decode(encode(src, src_mask), src_mask, tgt, tgt_mask);
}

// make_model实现
Transformer make_model(
    int src_vocab_size,
    int tgt_vocab_size,
    int N,
    int d_model,
    int d_ff,
    int h,
    float dropout,
    torch::Device device) {
    
    // 创建源语言和目标语言的嵌入层
    auto src_embed = Embeddings(d_model, src_vocab_size);
    auto tgt_embed = Embeddings(d_model, tgt_vocab_size);
    
    // 创建第一个编码器层作为模板（Encoder会创建N个独立的层）
    auto encoder_self_attn = MultiHeadedAttention(h, d_model, dropout);
    auto encoder_ff = PositionwiseFeedForward(d_model, d_ff, dropout);
    auto encoder_layer = EncoderLayer(d_model, encoder_self_attn, encoder_ff, dropout);
    
    // 创建第一个解码器层作为模板（Decoder会创建N个独立的层）
    auto decoder_self_attn = MultiHeadedAttention(h, d_model, dropout);
    auto decoder_src_attn = MultiHeadedAttention(h, d_model, dropout);
    auto decoder_ff = PositionwiseFeedForward(d_model, d_ff, dropout);
    auto decoder_layer = DecoderLayer(d_model, decoder_self_attn, decoder_src_attn, decoder_ff, dropout);
    
    // 创建编码器和解码器（内部会创建N个独立的层）
    auto encoder = Encoder(encoder_layer, N);
    auto decoder = Decoder(decoder_layer, N);
    
    // 创建位置编码（在指定设备上创建，确保pe buffer在GPU上）
    auto src_position = PositionalEncoding(d_model, dropout, 5000, device);
    auto tgt_position = PositionalEncoding(d_model, dropout, 5000, device);
    
    // 创建Transformer模型
    auto model = Transformer(
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        Generator(d_model, tgt_vocab_size),
        src_position,
        tgt_position
    );
    
    // 初始化模型参数（Xavier均匀初始化）
    for (auto& param : model->parameters()) {
        if (param.dim() > 1) {
            torch::nn::init::xavier_uniform_(param);
        }
    }
    
    // 将整个模型移动到指定设备（包括所有参数和buffer）
    // 这确保所有参数、buffer和子模块都在GPU上
    model->to(device);
    
    return model;
}

