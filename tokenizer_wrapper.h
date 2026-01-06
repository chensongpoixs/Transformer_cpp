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
 * 分词器包装器 (Tokenizer Wrapper)
 * 
 * 封装 SentencePiece 分词器的 C++ API，提供：
 * - 文本编码：将文本转换为 token ID 序列
 * - 文本解码：将 token ID 序列转换回文本
 * - 支持英文和中文分词器
 * 
 * 使用 SentencePiece 进行子词（subword）分词，能够处理未登录词（OOV）
				   
				   
				   
				   
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

#ifndef TRANSFORMER_TOKENIZER_WRAPPER_H
#define TRANSFORMER_TOKENIZER_WRAPPER_H

#include <string>
#include <vector>
#include <memory>

// 如果定义了USE_SENTENCEPIECE，使用真正的SentencePiece库
#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

/**
 * SentencePiece分词器包装类
 * 支持两种模式：
 * 1. 使用真正的SentencePiece库（定义USE_SENTENCEPIECE）
 * 2. 使用简化版本（fallback，基于字符的编码）
 */
class SentencePieceTokenizer {
public:
    SentencePieceTokenizer();
    ~SentencePieceTokenizer();
    
    /**
     * 加载分词器模型
     * @param model_path 模型文件路径（.model文件）
     * @return 是否加载成功
     */
    bool load(const std::string& model_path);
    
    /**
     * 将文本编码为token ID列表
     * @param text 输入文本
     * @return token ID列表
     */
    std::vector<int> encode_as_ids(const std::string& text);
    
    /**
     * 将token ID列表解码为文本
     * @param ids token ID列表
     * @return 解码后的文本
     */
    std::string decode_ids(const std::vector<int>& ids);
    
    /**
     * 获取PAD token ID
     */
    int pad_id() const { return pad_id_; }
    
    /**
     * 获取BOS token ID
     */
    int bos_id() const { return bos_id_; }
    
    /**
     * 获取EOS token ID
     */
    int eos_id() const { return eos_id_; }
    
    /**
     * 检查是否已加载
     */
    bool is_loaded() const { return loaded_; }

private:
    bool loaded_ = false;
    int pad_id_ = 0;
    int bos_id_ = 2;
    int eos_id_ = 3;
    
#ifdef USE_SENTENCEPIECE
    // 使用真正的SentencePiece处理器
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
#else
    // 简化模式：使用字符级编码
    void* processor_ = nullptr;
#endif
    
    // 简化模式的编码/解码方法
    std::vector<int> encode_simple(const std::string& text);
    std::string decode_simple(const std::vector<int>& ids);
};

/**
 * 加载英文分词器（使用默认路径）
 */
std::shared_ptr<SentencePieceTokenizer> english_tokenizer_load();

/**
 * 加载中文分词器（使用默认路径）
 */
std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load();

/**
 * 加载英文分词器（指定路径）
 * @param model_path 分词器模型文件路径
 */
std::shared_ptr<SentencePieceTokenizer> english_tokenizer_load(const std::string& model_path);

/**
 * 加载中文分词器（指定路径）
 * @param model_path 分词器模型文件路径
 */
std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load(const std::string& model_path);

#endif // TRANSFORMER_TOKENIZER_WRAPPER_H

