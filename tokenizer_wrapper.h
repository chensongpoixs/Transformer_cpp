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
 * 加载英文分词器
 */
std::shared_ptr<SentencePieceTokenizer> english_tokenizer_load();

/**
 * 加载中文分词器
 */
std::shared_ptr<SentencePieceTokenizer> chinese_tokenizer_load();

#endif // TRANSFORMER_TOKENIZER_WRAPPER_H

