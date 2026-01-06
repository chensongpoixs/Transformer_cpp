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
    auto tokenizer = std::make_shared<SentencePieceTokenizer>();
    // 尝试加载英文模型
    std::string model_path = "./tokenizer/eng.model";
    tokenizer->load(model_path);
    LOG_INFO(std::string("英文分词器初始化完成，模型: ") + model_path);
    return tokenizer;
}

std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load() {
    auto tokenizer = std::make_shared<SentencePieceTokenizer>();
    // 尝试加载中文模型
    std::string model_path = "./tokenizer/chn.model";
    tokenizer->load(model_path);
    LOG_INFO(std::string("中文分词器初始化完成，模型: ") + model_path);
    return tokenizer;
}

