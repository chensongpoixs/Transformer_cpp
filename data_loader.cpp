#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <torch/nn/utils/rnn.h>
#include "json.hpp"
#include "logger.h"

using namespace logging;
using json = nlohmann::json;

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
    LOG_INFO("开始加载数据集(JSON): " + data_path);

    std::ifstream file(data_path);
    if (!file.is_open()) {
        LOG_ERROR("无法打开数据文件: " + data_path);
        throw std::runtime_error("无法打开数据文件: " + data_path);
    }
    
    en_sentences.clear();
    cn_sentences.clear();

    try {
        json data;
        file >> data;
        file.close();

        if (!data.is_array()) {
            LOG_ERROR("JSON 格式错误: 根节点不是数组: " + data_path);
            throw std::runtime_error("JSON 格式错误: 根节点不是数组: " + data_path);
        }

        size_t entry_count = data.size();

        for (const auto& item : data) {
            // 期望格式: [ "<en sentence>", "<cn sentence>" ]
            if (!item.is_array() || item.size() < 2) {
                continue;
            }
            if (!item[0].is_string() || !item[1].is_string()) {
                continue;
            }
            std::string en = item[0].get<std::string>();
            std::string cn = item[1].get<std::string>();
            en_sentences.push_back(std::move(en));
            cn_sentences.push_back(std::move(cn));
        }

        std::ostringstream oss;
        oss << "数据文件加载完成(JSON): " << data_path
            << ", JSON条目数=" << entry_count
            << ", 有效样本数=" << en_sentences.size();
        LOG_INFO(oss.str());
    } catch (const std::exception& e) {
        LOG_ERROR(std::string("解析JSON失败: ") + data_path + ", 错误: " + e.what());
        throw;
    }
    
    // 按英文句子长度排序
    auto sorted_indices = len_argsort(en_sentences);
    std::vector<std::string> sorted_en, sorted_cn;
    for (auto idx : sorted_indices) {
        sorted_en.push_back(en_sentences[idx]);
        sorted_cn.push_back(cn_sentences[idx]);
    }
    en_sentences = sorted_en;
    cn_sentences = sorted_cn;

    {
        std::ostringstream oss;
        oss << "数据集排序完成, 总样本数=" << en_sentences.size();
        LOG_INFO(oss.str());
    }
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
    LOG_INFO("创建MTDataset, 数据路径: " + data_path);

    load_data(data_path);
    // 默认加载分词器
    sp_eng_ = english_tokenizer_load();
    sp_chn_ = chinese_tokenizer_load();
    PAD_ = sp_eng_->pad_id();
    BOS_ = sp_eng_->bos_id();
    EOS_ = sp_eng_->eos_id();

    {
        std::ostringstream oss;
        oss << "MTDataset 初始化完成: 样本数=" << en_sentences.size();
        LOG_INFO(oss.str());
    }
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

    {
        std::ostringstream oss;
        oss << "构建Batch: 样本数=" << batch_src_text.size()
            << ", 设备=" << (device.is_cuda() ? "CUDA" : "CPU");
        LOG_DEBUG(oss.str());

        // 打印前若干条原始文本，便于调试
        size_t print_count = std::min<size_t>(batch_src_text.size(), 3);
        for (size_t i = 0; i < print_count; ++i) {
            std::ostringstream oss_sample;
            oss_sample << "  样本[" << i << "]: src=\"" << batch_src_text[i]
                       << "\", trg=\"" << batch_trg_text[i] << "\"";
            LOG_DEBUG(oss_sample.str());
        }
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

    {
        std::ostringstream oss;
        oss << "Batch 序列长度: src_max_len=" << max_src_len
            << ", tgt_max_len=" << max_tgt_len;
        LOG_DEBUG(oss.str());
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

    {
        std::ostringstream oss;
        oss << "Batch 构建完成: src形状=[" << src.size(0) << ", " << src.size(1)
            << "], trg形状=[" << trg.size(0) << ", " << trg.size(1) << "]";
        LOG_DEBUG(oss.str());
    }

    return Batch(batch_src_text, batch_trg_text, src, trg, pad_idx, device);
}

