#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <torch/nn/utils/rnn.h>

// 简化版JSON解析（实际应该使用nlohmann/json库）
// 这里使用简单的文本解析作为占位符

// Batch实现
Batch::Batch(const std::vector<std::string>& src_text,
             const std::vector<std::string>& trg_text,
             torch::Tensor src,
             torch::Tensor trg,
             int pad,
             torch::Device device)
    : src_text(src_text), trg_text(trg_text) {
    
    // 移动到指定设备
    this->src = src.to(device);
    
    // 创建源语言mask
    this->src_mask = (this->src != pad).unsqueeze(-2).to(device);
    
    if (trg.defined()) {
        this->trg = trg.to(device);
        // decoder输入：去掉最后一个token
        this->trg = this->trg.slice(1, 0, -1);
        // decoder输出：从第二个token开始
        this->trg_y = this->trg.slice(1, 1);
        
        // 创建目标语言mask（包含subsequent mask）
        auto tgt_mask = (this->trg != pad).unsqueeze(-2).to(device);
        auto sub_mask = subsequent_mask(this->trg.size(-1), device);
        this->trg_mask = tgt_mask & sub_mask;
        
        // 计算有效token数量
        this->ntokens = (this->trg_y != pad).sum().item<int64_t>();
    } else {
        this->ntokens = 0;
    }
}

// MTDataset实现
void MTDataset::load_data(const std::string& data_path) {
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开数据文件: " + data_path);
    }
    
    en_sentences.clear();
    cn_sentences.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        // 简化版JSON解析：查找 ["...", "..."] 格式
        // 实际应该使用完整的JSON解析库（如nlohmann/json）
        size_t first_quote = line.find('"');
        if (first_quote == std::string::npos) continue;
        
        size_t second_quote = line.find('"', first_quote + 1);
        if (second_quote == std::string::npos) continue;
        
        std::string en = line.substr(first_quote + 1, second_quote - first_quote - 1);
        
        size_t third_quote = line.find('"', second_quote + 1);
        if (third_quote == std::string::npos) continue;
        
        size_t fourth_quote = line.find('"', third_quote + 1);
        if (fourth_quote == std::string::npos) continue;
        
        std::string cn = line.substr(third_quote + 1, fourth_quote - third_quote - 1);
        
        en_sentences.push_back(en);
        cn_sentences.push_back(cn);
    }
    
    file.close();
    
    // 按英文句子长度排序
    auto sorted_indices = len_argsort(en_sentences);
    std::vector<std::string> sorted_en, sorted_cn;
    for (auto idx : sorted_indices) {
        sorted_en.push_back(en_sentences[idx]);
        sorted_cn.push_back(cn_sentences[idx]);
    }
    en_sentences = sorted_en;
    cn_sentences = sorted_cn;
}

std::vector<size_t> MTDataset::len_argsort(const std::vector<std::string>& sentences) {
    std::vector<size_t> indices(sentences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&sentences](size_t a, size_t b) {
                  return sentences[a].length() < sentences[b].length();
              });
    return indices;
}

MTDataset::MTDataset(const std::string& data_path) {
    load_data(data_path);
    // 默认加载分词器
    sp_eng_ = english_tokenizer_load();
    sp_chn_ = chinese_tokenizer_load();
    PAD_ = sp_eng_->pad_id();
    BOS_ = sp_eng_->bos_id();
    EOS_ = sp_eng_->eos_id();
}

void MTDataset::set_tokenizers(std::shared_ptr<SentencePieceTokenizer> eng_tokenizer,
                               std::shared_ptr<SentencePieceTokenizer> chn_tokenizer) {
    sp_eng_ = eng_tokenizer;
    sp_chn_ = chn_tokenizer;
    if (sp_eng_) {
        PAD_ = sp_eng_->pad_id();
        BOS_ = sp_eng_->bos_id();
        EOS_ = sp_eng_->eos_id();
    }
}

Batch MTDataset::collate_fn(const std::vector<size_t>& indices,
                            torch::Device device,
                            int pad_idx, int bos_idx, int eos_idx,
                            int src_vocab_size, int tgt_vocab_size) const {
    std::vector<std::string> batch_src_text, batch_trg_text;
    
    // 收集文本
    for (auto idx : indices) {
        batch_src_text.push_back(en_sentences[idx]);
        batch_trg_text.push_back(cn_sentences[idx]);
    }
    
    // 使用SentencePiece分词器进行编码
    std::vector<std::vector<int64_t>> src_tokens_list, tgt_tokens_list;
    
    for (const auto& text : batch_src_text) {
        std::vector<int> ids = sp_eng_->encode_as_ids(text);
        std::vector<int64_t> token_ids;
        token_ids.push_back(bos_idx);
        for (int id : ids) {
            token_ids.push_back(id);
        }
        token_ids.push_back(eos_idx);
        src_tokens_list.push_back(token_ids);
    }
    
    for (const auto& text : batch_trg_text) {
        std::vector<int> ids = sp_chn_->encode_as_ids(text);
        std::vector<int64_t> token_ids;
        token_ids.push_back(bos_idx);
        for (int id : ids) {
            token_ids.push_back(id);
        }
        token_ids.push_back(eos_idx);
        tgt_tokens_list.push_back(token_ids);
    }
    
    // 找到最大长度
    int max_src_len = 0, max_tgt_len = 0;
    for (const auto& tokens : src_tokens_list) {
        max_src_len = std::max(max_src_len, static_cast<int>(tokens.size()));
    }
    for (const auto& tokens : tgt_tokens_list) {
        max_tgt_len = std::max(max_tgt_len, static_cast<int>(tokens.size()));
    }
    
    // 创建tensor并填充
    auto src = torch::full({static_cast<int64_t>(indices.size()), max_src_len}, 
                           pad_idx,
                           torch::TensorOptions().dtype(torch::kLong).device(device));
    auto trg = torch::full({static_cast<int64_t>(indices.size()), max_tgt_len},
                           pad_idx,
                           torch::TensorOptions().dtype(torch::kLong).device(device));
    
    for (size_t i = 0; i < src_tokens_list.size(); ++i) {
        for (size_t j = 0; j < src_tokens_list[i].size(); ++j) {
            src[i][j] = src_tokens_list[i][j];
        }
    }
    
    for (size_t i = 0; i < tgt_tokens_list.size(); ++i) {
        for (size_t j = 0; j < tgt_tokens_list[i].size(); ++j) {
            trg[i][j] = tgt_tokens_list[i][j];
        }
    }
    
    return Batch(batch_src_text, batch_trg_text, src, trg, pad_idx, device);
}

