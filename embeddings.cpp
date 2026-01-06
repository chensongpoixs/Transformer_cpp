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
 * 词嵌入和位置编码实现 (Embeddings and Positional Encoding Implementation)
 * 
 * 实现词嵌入层和位置编码层的前向传播
 * 
 * Embeddings: 将词索引转换为向量，并乘以 sqrt(d_model)
 * PositionalEncoding: 添加位置编码到嵌入向量
				   
				   
				   
				   
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

#include "embeddings.h"
#include <cmath>

// Embeddings实现
EmbeddingsImpl::EmbeddingsImpl(int d_model, int vocab_size)
    : embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)),
      d_model(d_model) {
    register_module("embedding", embedding);
}

torch::Tensor EmbeddingsImpl::forward(torch::Tensor x) {
    // 返回embedding矩阵乘以sqrt(d_model)以保持方差
    return embedding->forward(x) * std::sqrt(static_cast<float>(d_model));
}

// PositionalEncoding实现
PositionalEncodingImpl::PositionalEncodingImpl(int d_model, float drop_rate, int max_len, torch::Device device)
    : dropout(torch::nn::DropoutOptions(drop_rate)) {
    register_module("dropout", dropout);
    
    // 生成位置索引 [max_len, 1]，直接在指定设备上创建
    auto position = torch::arange(0, max_len, torch::TensorOptions().dtype(torch::kFloat32).device(device)).unsqueeze(1);
    
    // 计算div_term: exp(-log(10000) / d_model * 2i)
    // 对于偶数维度: i = 0, 2, 4, ... -> div_term shape: [d_model/2]
    auto div_term = torch::exp(
        torch::arange(0, d_model, 2, torch::TensorOptions().dtype(torch::kFloat32).device(device)) * 
        (-std::log(10000.0f) / static_cast<float>(d_model))
    );
    
    // 计算位置编码: position * div_term shape: [max_len, d_model/2]
    auto pos_encoding = position * div_term;
    
    // 分别计算sin和cos值
    auto sin_vals = torch::sin(pos_encoding);  // [max_len, d_model/2]
    auto cos_vals = torch::cos(pos_encoding);  // [max_len, d_model/2]
    
    // 构建完整的pe矩阵：交替使用sin和cos值
    // 初始化pe为全零矩阵 [max_len, d_model]
    pe = torch::zeros({max_len, d_model}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // 使用index_put_来正确赋值
    // 为偶数维度（0, 2, 4, ...）赋值sin值
    // 为奇数维度（1, 3, 5, ...）赋值cos值
    int num_pairs = d_model / 2;
    for (int i = 0; i < num_pairs; ++i) {
        int even_idx = i * 2;
        int odd_idx = i * 2 + 1;
        
        // 使用index_put_为偶数维度赋值sin值
        pe.index_put_({torch::indexing::Slice(), even_idx}, sin_vals.select(1, i));
        
        // 使用index_put_为奇数维度赋值cos值（如果存在）
        if (odd_idx < d_model) {
            pe.index_put_({torch::indexing::Slice(), odd_idx}, cos_vals.select(1, i));
        }
    }
    
    // 如果d_model是奇数，最后一个维度使用sin
    if (d_model % 2 == 1 && num_pairs > 0) {
        pe.index_put_({torch::indexing::Slice(), d_model - 1}, sin_vals.select(1, num_pairs - 1));
    }
    
    // 增加batch维度: [1, max_len, d_model]
    pe = pe.unsqueeze(0);
    
    // 注册为buffer（不参与训练的参数），确保在正确的设备上
    register_buffer("pe", pe);
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) {
    // 将位置编码添加到输入
    // pe的形状应该是 [1, max_len, d_model]
    // x的形状是 [batch_size, seq_len, d_model]
    // Python版本: x = x + self.pe[:, :x.size(1)]
    // 这相当于取pe的 [:, 0:seq_len, :]，得到 [1, seq_len, d_model]
    // 然后通过广播机制与 [batch_size, seq_len, d_model] 相加
    
    // 获取序列长度
    int seq_len = x.size(1);
    
    // 取pe的前seq_len个位置: pe[:, 0:seq_len, :]
    // pe的形状: [1, max_len, d_model]
    // pe_slice的形状: [1, seq_len, d_model]
    auto pe_slice = pe.slice(1, 0, seq_len);
    
    // 确保pe_slice和x在同一个设备上
    if (pe_slice.device() != x.device()) {
        pe_slice = pe_slice.to(x.device());
    }
    
    // 广播相加: [batch_size, seq_len, d_model] + [1, seq_len, d_model]
    // LibTorch会自动处理广播
    x = x + pe_slice;
    return dropout->forward(x);
}

