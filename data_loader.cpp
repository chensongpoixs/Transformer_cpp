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
 * 数据加载器实现 (Data Loader Implementation)
 * 
 * 实现机器翻译数据集的加载和批处理：
 * - 从 JSON 文件加载训练数据
 * - 使用 SentencePiece 进行分词
 * - 构建批次，处理 padding 和 mask
 * - 支持按长度排序的 bucket 采样
				   
				   
				   
				   
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
    
    //torch::util::crc64();
    
    // 移动到指定设备
    this->src = src.to(device);
    
    // 创建源语言mask
    this->src_mask = (this->src != pad).unsqueeze(-2).to(device);
    
    if (trg.defined()) {
        // 先在设备上保存完整的trg，然后基于完整trg构造 decoder 输入和输出
        auto full_trg = trg.to(device);              // 形状: [batch, L]
        // decoder输入：去掉最后一个token => [batch, L-1]
        this->trg = full_trg.slice(1, 0, -1);
        // decoder输出：从第二个token开始 => [batch, L-1]
        this->trg_y = full_trg.slice(1, 1);
        
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
    en_sentences = std::move(sorted_en);
    cn_sentences = std::move(sorted_cn);

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

// 根据句子长度生成排序后的索引（用于bucket采样）
std::vector<size_t> MTDataset::make_length_sorted_indices() const {
    return len_argsort(en_sentences);
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
        //std::ostringstream oss;
        //oss << "构建Batch: 样本数=" << batch_src_text.size()
        //    << ", 设备=" << (device.is_cuda() ? "CUDA" : "CPU");
        //LOG_DEBUG(oss.str());

        //// 打印前若干条原始文本，便于调试
        //size_t print_count = std::min<size_t>(batch_src_text.size(), 3);
        //for (size_t i = 0; i < print_count; ++i) {
        //    std::ostringstream oss_sample;
        //    oss_sample << "  样本[" << i << "]: src=\"" << batch_src_text[i]
        //               << "\", trg=\"" << batch_trg_text[i] << "\"";
        //    LOG_DEBUG(oss_sample.str());
        //}
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

   /* {
        std::ostringstream oss;
        oss << "Batch 序列长度: src_max_len=" << max_src_len
            << ", tgt_max_len=" << max_tgt_len;
        LOG_DEBUG(oss.str());
    }*/
    
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

   /* {
        std::ostringstream oss;
        oss << "Batch 构建完成: src形状=[" << src.size(0) << ", " << src.size(1)
            << "], trg形状=[" << trg.size(0) << ", " << trg.size(1) << "]";
        LOG_DEBUG(oss.str());
    }*/

    return Batch(batch_src_text, batch_trg_text, src, trg, pad_idx, device);
}

