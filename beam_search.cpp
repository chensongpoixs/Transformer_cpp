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
 * Beam Search 解码实现 (Beam Search Decoding Implementation)
 * 
 * 实现束搜索算法，用于生成翻译结果
 * 
 * 算法流程：
 * 1. 初始化 beam，包含 BOS token
 * 2. 对每个时间步扩展候选并保留 top-k
 * 3. 当遇到 EOS token 时完成该 beam
 * 4. 返回所有完成的序列及其分数
				   
				   
				   
				   
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

#include "beam_search.h"
#include <algorithm>
#include <map>
#include <numeric>

// Beam实现
Beam::Beam(int size, int pad, int bos, int eos, torch::Device device)
    : size_(size), done_(false), PAD_(pad), BOS_(bos), EOS_(eos), device_(device) {
    
    // 初始化分数为0
    scores_ = torch::zeros({size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // 初始化输出：第一个位置是BOS，其他是PAD
    auto first_ys = torch::full({size}, PAD_, torch::TensorOptions().dtype(torch::kLong).device(device));
    first_ys[0] = BOS_;
    next_ys_.push_back(first_ys);
}

torch::Tensor Beam::get_current_state() {
    return get_tentative_hypothesis();
}

torch::Tensor Beam::get_current_origin() {
    if (prev_ks_.empty()) {
        return torch::zeros({size_}, torch::TensorOptions().dtype(torch::kLong).device(device_));
    }
    return prev_ks_.back();
}

bool Beam::advance(torch::Tensor word_logprob) {
    // word_logprob: [beam_size, vocab_size] 或 [vocab_size]（初始情况）
    
    int num_words = word_logprob.size(-1);
    
    torch::Tensor beam_lk;
    if (prev_ks_.size() > 0) {
        // 累加之前的分数
        beam_lk = word_logprob + scores_.unsqueeze(-1).expand_as(word_logprob);
    } else {
        // 初始情况：只使用第一个beam的分数
        beam_lk = word_logprob[0];
    }
    
    // 展平并获取top-k
    auto flat_beam_lk = beam_lk.view(-1);
    auto topk_result = flat_beam_lk.topk(size_, 0, true, true);
    auto best_scores = std::get<0>(topk_result);
    auto best_scores_id = std::get<1>(topk_result);
    
    all_scores_.push_back(scores_);
    scores_ = best_scores;
    
    // 计算每个分数来自哪个beam和哪个词
    auto prev_k = best_scores_id / num_words;
    prev_ks_.push_back(prev_k);
    
    auto next_y = best_scores_id - prev_k * num_words;
    next_ys_.push_back(next_y);
    
    // 检查是否完成（top beam以EOS结束）
    if (next_ys_.back()[0].item<int>() == EOS_) {
        done_ = true;
        all_scores_.push_back(scores_);
    }
    
    return done_;
}

std::pair<torch::Tensor, torch::Tensor> Beam::sort_scores() {
    auto sorted = torch::sort(scores_, 0, true);
    return {std::get<0>(sorted), std::get<1>(sorted)};
}

std::pair<float, int> Beam::get_the_best_score_and_idx() {
    auto [scores, ids] = sort_scores();
    return {scores[1].item<float>(), ids[1].item<int>()};
}

torch::Tensor Beam::get_tentative_hypothesis() {
    if (next_ys_.size() == 1) {
        return next_ys_[0].unsqueeze(-1);
    }
    
    auto [scores, keys] = sort_scores();
    std::vector<std::vector<int>> hyps;
    
    for (int i = 0; i < size_; ++i) {
        int k = keys[i].item<int>();
        auto hyp = get_hypothesis(k);
        hyp.insert(hyp.begin(), BOS_);
        hyps.push_back(hyp);
    }
    
    // 转换为tensor
    int max_len = 0;
    for (const auto& hyp : hyps) {
        max_len = std::max(max_len, static_cast<int>(hyp.size()));
    }
    
    auto dec_seq = torch::full({size_, max_len}, PAD_, 
                               torch::TensorOptions().dtype(torch::kLong).device(device_));
    for (int i = 0; i < size_; ++i) {
        for (size_t j = 0; j < hyps[i].size(); ++j) {
            dec_seq[i][j] = hyps[i][j];
        }
    }
    
    return dec_seq;
}

std::vector<int> Beam::get_hypothesis(int k) {
    std::vector<int> hyp;
    
    // 回溯构建假设
    int current_k = k;
    for (int j = static_cast<int>(prev_ks_.size()) - 1; j >= 0; --j) {
        hyp.push_back(next_ys_[j + 1][current_k].item<int>());
        current_k = prev_ks_[j][current_k].item<int>();
    }
    
    // 反转（因为是从后往前构建的）
    std::reverse(hyp.begin(), hyp.end());
    return hyp;
}

// Beam Search实现
std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<float>>> 
beam_search(Transformer model,
            torch::Tensor src,
            torch::Tensor src_mask,
            int max_len,
            int pad,
            int bos,
            int eos,
            int beam_size,
            torch::Device device) {
    
    torch::NoGradGuard no_grad;
    model->eval();
    
    // 编码源语言
    auto src_enc = model->encode(src, src_mask);
    
    // 为beam search重复数据
    int batch_size = src_enc.size(0);
    int sent_len = src_enc.size(1);
    int h_dim = src_enc.size(2);
    
    // 扩展src_enc和src_mask以支持beam search
    src_enc = src_enc.unsqueeze(1).repeat({1, beam_size, 1, 1})
                     .view({batch_size * beam_size, sent_len, h_dim});
    src_mask = src_mask.unsqueeze(1).repeat({1, beam_size, 1, 1})
                       .view({batch_size * beam_size, 1, src_mask.size(-1)});
    
    // 创建beams
    std::vector<std::unique_ptr<Beam>> inst_dec_beams;
    for (int i = 0; i < batch_size; ++i) {
        inst_dec_beams.push_back(std::make_unique<Beam>(beam_size, pad, bos, eos, device));
    }
    
    // 活跃实例索引
    std::vector<int> active_inst_idx_list(batch_size);
    std::iota(active_inst_idx_list.begin(), active_inst_idx_list.end(), 0);
    
    // 实例索引到tensor位置的映射
    std::map<int, int> inst_idx_to_position_map;
    for (size_t i = 0; i < active_inst_idx_list.size(); ++i) {
        inst_idx_to_position_map[active_inst_idx_list[i]] = i;
    }
    
    // 解码循环
    for (int len_dec_seq = 1; len_dec_seq <= max_len; ++len_dec_seq) {
        // 准备解码序列
        std::vector<torch::Tensor> dec_partial_seqs;
        for (int idx : active_inst_idx_list) {
            if (!inst_dec_beams[idx]->done()) {
                dec_partial_seqs.push_back(inst_dec_beams[idx]->get_current_state());
            }
        }
        
        if (dec_partial_seqs.empty()) {
            break;
        }
        
        // 堆叠并重塑
        auto dec_seq = torch::stack(dec_partial_seqs, 0).to(device);
        int n_active_inst = dec_seq.size(0);
        dec_seq = dec_seq.view({n_active_inst * beam_size, len_dec_seq});
        
        // 扩展src_enc和src_mask以匹配
        auto active_src_enc = src_enc.narrow(0, 0, n_active_inst * beam_size);
        auto active_src_mask = src_mask.narrow(0, 0, n_active_inst * beam_size);
        
        // 创建tgt_mask
        auto tgt_mask = subsequent_mask(len_dec_seq, device).to(dec_seq.device());
        
        // 解码
        auto out = model->decode(active_src_enc, active_src_mask, dec_seq, tgt_mask);
        
        // 获取最后一个位置的输出并通过generator
        auto last_out = out.select(1, -1);  // [n_active_inst * beam_size, d_model]
        auto word_logprob = model->get_generator()->forward(last_out);  // [n_active_inst * beam_size, vocab_size]
        word_logprob = word_logprob.view({n_active_inst, beam_size, -1});
        
        // 更新beams
        std::vector<int> new_active_inst_idx_list;
        for (int inst_idx : active_inst_idx_list) {
            if (inst_dec_beams[inst_idx]->done()) {
                continue;
            }
            
            int inst_position = inst_idx_to_position_map[inst_idx];
            bool is_complete = inst_dec_beams[inst_idx]->advance(word_logprob[inst_position]);
            
            if (!is_complete) {
                new_active_inst_idx_list.push_back(inst_idx);
            }
        }
        
        active_inst_idx_list = new_active_inst_idx_list;
        
        if (active_inst_idx_list.empty()) {
            break;
        }
        
        // 更新映射
        inst_idx_to_position_map.clear();
        for (size_t i = 0; i < active_inst_idx_list.size(); ++i) {
            inst_idx_to_position_map[active_inst_idx_list[i]] = i;
        }
        
        // 更新src_enc和src_mask（只保留活跃的）
        if (active_inst_idx_list.size() < static_cast<size_t>(n_active_inst)) {
            std::vector<int64_t> indices;
            for (int idx : active_inst_idx_list) {
                int pos = inst_idx_to_position_map[idx];
                for (int b = 0; b < beam_size; ++b) {
                    indices.push_back(pos * beam_size + b);
                }
            }
            auto indices_tensor = torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong).device(device));
            src_enc = src_enc.index_select(0, indices_tensor);
            src_mask = src_mask.index_select(0, indices_tensor);
        }
    }
    
    // 收集结果
    std::vector<std::vector<std::vector<int>>> batch_hyp;
    std::vector<std::vector<float>> batch_scores;
    
    for (int i = 0; i < batch_size; ++i) {
        auto [scores, tail_idxs] = inst_dec_beams[i]->sort_scores();
        
        std::vector<std::vector<int>> hyps;
        std::vector<float> scs;
        
        int n_best = std::min(beam_size, static_cast<int>(tail_idxs.size(0)));
        for (int j = 0; j < n_best; ++j) {
            int idx = tail_idxs[j].item<int>();
            hyps.push_back(inst_dec_beams[i]->get_hypothesis(idx));
            scs.push_back(scores[j].item<float>());
        }
        
        batch_hyp.push_back(hyps);
        batch_scores.push_back(scs);
        
        // 显式释放 scores 和 tail_idxs 张量
        scores = torch::Tensor();
        tail_idxs = torch::Tensor();
    }
    
    // 显式释放所有中间张量
    src_enc = torch::Tensor();
    src_mask = torch::Tensor();
    
    // 清理Beam对象（会自动释放其持有的张量）
    inst_dec_beams.clear();
    
    // 清理CUDA缓存
    if (device.is_cuda()) {
    //    c10::cuda::CUDACachingAllocator::emptyCache();
    }
    
    return {batch_hyp, batch_scores};
}

