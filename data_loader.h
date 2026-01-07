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
 * 数据加载器 (Data Loader)
 * 
 * 实现机器翻译数据集加载和批处理功能：
 * - MTDataset: 数据集类，从 JSON 文件加载训练数据
 * - Batch: 批次数据结构，包含源语言和目标语言的 token 序列和 mask
 * - collate_fn: 将多个样本组合成批次，处理 padding 和 mask
 * - make_length_sorted_indices: 生成按长度排序的索引，用于 bucket 采样
				   
				   
				   
				   
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

#ifndef TRANSFORMER_DATA_LOADER_H
#define TRANSFORMER_DATA_LOADER_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include "utils.h"
#include "tokenizer_wrapper.h"
//#include <torch/csrc/python_headers.h>
//#include <torch/csrc/DataLoader.h>
/**
 * Batch类
 * 存储一个训练批次的数据
 */
struct Batch {
    std::vector<std::string> src_text;  // 源语言文本
    std::vector<std::string> trg_text;  // 目标语言文本
    torch::Tensor src;                  // 源语言token序列
    torch::Tensor trg;                  // 目标语言token序列（输入）
    torch::Tensor trg_y;                // 目标语言token序列（输出）
    torch::Tensor src_mask;             // 源语言mask
    torch::Tensor trg_mask;             // 目标语言mask
    int64_t ntokens;                    // 有效token数量
    
    // 默认构造函数（允许声明未初始化的 Batch 对象）
    Batch() = default;
    
    // 带参数的构造函数
    Batch(const std::vector<std::string>& src_text,
          const std::vector<std::string>& trg_text,
          torch::Tensor src,
          torch::Tensor trg,
          int pad,
          torch::Device device);
};

/**
 * 数据集类（简化版）
 * 注意：完整实现需要集成SentencePiece分词器
 */
class MTDataset {
public:
    MTDataset(const std::string& data_path);
    
    // 获取数据集大小
    size_t size() const { return en_sentences.size(); }
    
    // 获取指定索引的数据
    std::pair<std::string, std::string> get(size_t idx) const {
        return {en_sentences[idx], cn_sentences[idx]};
    }
    
    // 创建batch（使用SentencePiece分词器，支持批量分词优化）
    Batch collate_fn(const std::vector<size_t>& indices, 
                     torch::Device device,
                     int pad_idx, int bos_idx, int eos_idx,
                     int src_vocab_size, int tgt_vocab_size) const;

    // 根据句子长度生成排序后的索引（用于bucket采样）
    std::vector<size_t> make_length_sorted_indices() const;
    
    // 设置分词器
    void set_tokenizers(std::shared_ptr<SentencePieceTokenizer> eng_tokenizer,
                       std::shared_ptr<SentencePieceTokenizer> chn_tokenizer);

private:
    std::vector<std::string> en_sentences;
    std::vector<std::string> cn_sentences;
    
    std::shared_ptr<SentencePieceTokenizer> sp_eng_;
    std::shared_ptr<SentencePieceTokenizer> sp_chn_;
    int PAD_;
    int BOS_;
    int EOS_;
    
    void load_data(const std::string& data_path);
    static std::vector<size_t> len_argsort(const std::vector<std::string>& sentences);
};

#endif // TRANSFORMER_DATA_LOADER_H

