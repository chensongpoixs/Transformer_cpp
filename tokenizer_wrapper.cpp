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
 * 分词器包装器实现 (Tokenizer Wrapper Implementation)
 * 
 * 封装 SentencePiece 分词器的 C++ API
 * 
 * 功能：
 * - 加载 SentencePiece 模型
 * - 文本编码和解码
 * - 支持英文和中文分词器
				   
				   
				   
				   
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

#include "tokenizer_wrapper.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include "logger.h"

using namespace logging;


#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

SentencePieceTokenizer::SentencePieceTokenizer() {
#ifdef USE_SENTENCEPIECE
    processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
#endif
}

SentencePieceTokenizer::~SentencePieceTokenizer() {
    // unique_ptr会自动清理
}

bool SentencePieceTokenizer::load(const std::string& model_path) {
#ifdef USE_SENTENCEPIECE
    // 使用真正的SentencePiece库
    if (!processor_) {
        processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    }
    
    // SentencePiece的Load方法返回util::Status
    auto status = processor_->Load(model_path);
    if (!status.ok()) {
        LOG_ERROR(std::string("无法加载SentencePiece模型: ") + model_path +
                  ", 错误: " + status.ToString());
        loaded_ = false;
        return false;
    }
    
    // 从模型获取特殊token ID
    pad_id_ = processor_->pad_id();
    bos_id_ = processor_->bos_id();
    eos_id_ = processor_->eos_id();
    
    loaded_ = true;
    LOG_INFO(std::string("SentencePiece 模型加载成功: ") + model_path +
             ", pad_id=" + std::to_string(pad_id_) +
             ", bos_id=" + std::to_string(bos_id_) +
             ", eos_id=" + std::to_string(eos_id_));
    return true;
#else
    // 简化模式：检查文件是否存在
    std::ifstream file(model_path);
    if (!file.is_open()) {
        LOG_WARN(std::string("SentencePiece 模型文件不存在: ") + model_path +
                 ", 使用字符级简化模式");
        loaded_ = true;  // 标记为已加载，但使用简化模式
        return true;
    }
    file.close();
    
    // 简化模式：标记为已加载
    loaded_ = true;
    return true;
#endif
}

std::vector<int> SentencePieceTokenizer::encode_as_ids(const std::string& text) {
    if (!loaded_) {
        return {};
    }
    
#ifdef USE_SENTENCEPIECE
    // 使用真正的SentencePiece库
    if (processor_) {
        std::vector<int> ids;
        // SentencePiece的Encode方法返回void，通过引用参数返回结果
        processor_->Encode(text, &ids);
        return ids;
    }
#endif
    
    // 简化模式：使用字符级编码
    return encode_simple(text);
}

std::vector<std::vector<int>> SentencePieceTokenizer::encode_as_ids_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> result;
    if (!loaded_) {
        return result;
    }
    
#ifdef USE_SENTENCEPIECE
    // 使用真正的SentencePiece库：逐条调用 Encode，保持接口兼容性
    if (processor_) {
        result.reserve(texts.size());
        for (const auto& t : texts) {
            std::vector<int> ids;
            auto status = processor_->Encode(t, &ids);
            if (!status.ok()) {
                LOG_WARN(std::string("SentencePiece 批量编码单条文本失败: ") + status.ToString());
                result.emplace_back();  // 保持对齐，推入空结果
            } else {
                result.push_back(std::move(ids));
            }
        }
        return result;
    }
#endif
    // 简化模式或未初始化 SentencePiece 时，逐条调用 encode_as_ids
    result.reserve(texts.size());
    for (const auto& t : texts) {
        result.push_back(encode_as_ids(t));
    }
    return result;
}

std::string SentencePieceTokenizer::decode_ids(const std::vector<int>& ids) {
    if (!loaded_) {
        return "";
    }
    
#ifdef USE_SENTENCEPIECE
    // 使用真正的SentencePiece库
    if (processor_) {
        std::string text;
        // SentencePiece的Decode方法返回void，通过引用参数返回结果
        processor_->Decode(ids, &text);
        return text;
    }
#endif
    
    // 简化模式：使用字符级解码
    return decode_simple(ids);
}

// 简化模式的编码方法
std::vector<int> SentencePieceTokenizer::encode_simple(const std::string& text) {
    std::vector<int> ids;
    ids.reserve(text.length());
    
    // 简单的字符到ID映射（仅用于演示）
    // 实际应该使用训练好的SentencePiece模型
    for (char c : text) {
        int id = static_cast<unsigned char>(c);
        // 限制在合理范围内（避免特殊字符）
        if (id >= 32 && id < 127) {
            ids.push_back(id);
        }
    }
    
    return ids;
}

// 简化模式的解码方法
std::string SentencePieceTokenizer::decode_simple(const std::vector<int>& ids) {
    std::string text;
    text.reserve(ids.size());
    
    for (int id : ids) {
        // 过滤掉特殊token
        if (id == pad_id_ || id == bos_id_ || id == eos_id_) {
            continue;
        }
        // 限制在ASCII可打印字符范围
        if (id >= 32 && id < 127) {
            text += static_cast<char>(id);
        }
    }
    
    return text;
}

std::shared_ptr<SentencePieceTokenizer> english_tokenizer_load() {
    return english_tokenizer_load("./tokenizer/eng.model");
}

std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load() {
    return chinese_tokenizer_load("./tokenizer/chn.model");
}

std::shared_ptr<SentencePieceTokenizer> english_tokenizer_load(const std::string& model_path) {
    auto tokenizer = std::make_shared<SentencePieceTokenizer>();
    // 尝试加载英文模型
    tokenizer->load(model_path);
    LOG_INFO(std::string("英文分词器初始化完成，模型: ") + model_path);
    return tokenizer;
}

std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load(const std::string& model_path) {
    auto tokenizer = std::make_shared<SentencePieceTokenizer>();
    // 尝试加载中文模型
    tokenizer->load(model_path);
    LOG_INFO(std::string("中文分词器初始化完成，模型: ") + model_path);
    return tokenizer;
}

