#ifndef TRANSFORMER_DATA_LOADER_H
#define TRANSFORMER_DATA_LOADER_H

#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include "utils.h"
#include "tokenizer_wrapper.h"

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
    
    // 创建batch（使用SentencePiece分词器）
    Batch collate_fn(const std::vector<size_t>& indices, 
                     torch::Device device,
                     int pad_idx, int bos_idx, int eos_idx,
                     int src_vocab_size, int tgt_vocab_size) const;
    
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

