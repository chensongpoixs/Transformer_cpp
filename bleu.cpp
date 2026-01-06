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
 * BLEU 分数计算实现 (BLEU Score Calculation Implementation)
 * 
 * 实现基于 n-gram 精确度的 BLEU 分数计算
 * 
 * 计算流程：
 * 1. 计算 1-gram 到 N-gram 的精确度
 * 2. 应用长度惩罚（brevity penalty）
 * 3. 计算几何平均得到最终 BLEU 分数
				   
				   
				   
				   
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

#include "bleu.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>

// 生成n-gram
std::vector<std::vector<std::string>> get_ngrams(const std::vector<std::string>& tokens, int n) {
    std::vector<std::vector<std::string>> ngrams;
    if (tokens.size() < static_cast<size_t>(n)) {
        return ngrams;
    }
    
    for (size_t i = 0; i <= tokens.size() - n; ++i) {
        std::vector<std::string> ngram;
        for (int j = 0; j < n; ++j) {
            ngram.push_back(tokens[i + j]);
        }
        ngrams.push_back(ngram);
    }
    
    return ngrams;
}

// 计算n-gram计数
std::map<std::vector<std::string>, int> count_ngrams(const std::vector<std::string>& tokens, int n) {
    std::map<std::vector<std::string>, int> counts;
    auto ngrams = get_ngrams(tokens, n);
    for (const auto& ngram : ngrams) {
        counts[ngram]++;
    }
    return counts;
}

float compute_precision(const std::vector<std::string>& candidate,
                        const std::vector<std::string>& reference,
                        int n) {
    auto cand_ngrams = count_ngrams(candidate, n);
    auto ref_ngrams = count_ngrams(reference, n);
    
    int matches = 0;
    int total = 0;
    
    for (const auto& [ngram, count] : cand_ngrams) {
        total += count;
        if (ref_ngrams.find(ngram) != ref_ngrams.end()) {
            matches += std::min(count, ref_ngrams[ngram]);
        }
    }
    
    if (total == 0) return 0.0f;
    return static_cast<float>(matches) / total;
}

float compute_modified_precision(const std::vector<std::vector<std::string>>& references,
                                 const std::vector<std::string>& candidate,
                                 int n) {
    // 计算所有参考的n-gram计数（取最大值）
    std::map<std::vector<std::string>, int> max_ref_counts;
    
    for (const auto& ref : references) {
        auto ref_ngrams = count_ngrams(ref, n);
        for (const auto& [ngram, count] : ref_ngrams) {
            max_ref_counts[ngram] = std::max(max_ref_counts[ngram], count);
        }
    }
    
    // 计算候选的n-gram计数
    auto cand_ngrams = count_ngrams(candidate, n);
    
    // 计算匹配数（clip计数）
    int matches = 0;
    int total = 0;
    
    for (const auto& [ngram, count] : cand_ngrams) {
        total += count;
        if (max_ref_counts.find(ngram) != max_ref_counts.end()) {
            matches += std::min(count, max_ref_counts[ngram]);
        }
    }
    
    if (total == 0) return 0.0f;
    return static_cast<float>(matches) / total;
}

float brevity_penalty(int candidate_len, int closest_ref_len) {
    if (candidate_len > closest_ref_len) {
        return 1.0f;
    }
    return std::exp(1.0f - static_cast<float>(closest_ref_len) / candidate_len);
}

int closest_ref_length(const std::vector<std::vector<std::string>>& references, int candidate_len) {
    int closest = references[0].size();
    int min_diff = std::abs(static_cast<int>(references[0].size()) - candidate_len);
    
    for (const auto& ref : references) {
        int diff = std::abs(static_cast<int>(ref.size()) - candidate_len);
        if (diff < min_diff) {
            min_diff = diff;
            closest = ref.size();
        }
    }
    
    return closest;
}

float compute_bleu(const std::vector<std::string>& candidate,
                   const std::vector<std::vector<std::string>>& references,
                   int max_n) {
    if (candidate.empty()) {
        return 0.0f;
    }
    
    // 计算各n-gram的精确度
    std::vector<float> precisions;
    for (int n = 1; n <= max_n; ++n) {
        float prec = compute_modified_precision(references, candidate, n);
        precisions.push_back(prec);
    }
    
    // 如果任何精确度为0，返回0
    if (std::any_of(precisions.begin(), precisions.end(), 
                    [](float p) { return p == 0.0f; })) {
        return 0.0f;
    }
    
    // 计算几何平均
    float geo_mean = 1.0f;
    for (float prec : precisions) {
        geo_mean *= prec;
    }
    geo_mean = std::pow(geo_mean, 1.0f / max_n);
    
    // 计算brevity penalty
    int candidate_len = candidate.size();
    int closest_ref_len = closest_ref_length(references, candidate_len);
    float bp = brevity_penalty(candidate_len, closest_ref_len);
    
    // BLEU分数
    return bp * geo_mean * 100.0f;
}

float corpus_bleu(const std::vector<std::vector<std::string>>& candidates,
                  const std::vector<std::vector<std::vector<std::string>>>& references,
                  int max_n) {
    if (candidates.size() != references.size()) {
        return 0.0f;
    }
    
    // 语料库级别的n-gram计数
    std::vector<std::map<std::vector<std::string>, int>> cand_ngram_counts(max_n);
    std::vector<std::map<std::vector<std::string>, int>> ref_ngram_counts(max_n);
    
    int total_cand_ngrams = 0;
    int total_matches = 0;
    int total_cand_len = 0;
    int total_ref_len = 0;
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        const auto& refs = references[i];
        
        total_cand_len += candidate.size();
        
        // 找到最近的参考长度
        int closest_ref_len = closest_ref_length(refs, candidate.size());
        total_ref_len += closest_ref_len;
        
        // 计算各n-gram的匹配
        for (int n = 1; n <= max_n; ++n) {
            auto cand_ngrams = count_ngrams(candidate, n);
            total_cand_ngrams += cand_ngrams.size();
            
            // 计算所有参考的最大计数
            std::map<std::vector<std::string>, int> max_ref_ngrams;
            for (const auto& ref : refs) {
                auto ref_ngrams = count_ngrams(ref, n);
                for (const auto& [ngram, count] : ref_ngrams) {
                    max_ref_ngrams[ngram] = std::max(max_ref_ngrams[ngram], count);
                }
            }
            
            // 计算匹配数
            for (const auto& [ngram, count] : cand_ngrams) {
                if (max_ref_ngrams.find(ngram) != max_ref_ngrams.end()) {
                    total_matches += std::min(count, max_ref_ngrams[ngram]);
                }
            }
        }
    }
    
    // 计算精确度
    if (total_cand_ngrams == 0) {
        return 0.0f;
    }
    
    float precision = static_cast<float>(total_matches) / total_cand_ngrams;
    
    // 计算brevity penalty
    float bp = brevity_penalty(total_cand_len, total_ref_len);
    
    // BLEU分数（简化版，实际应该分别计算各n-gram的精确度然后取几何平均）
    return bp * precision * 100.0f;
}

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> tokenize_chinese(const std::string& text) {
    std::vector<std::string> tokens;
    // 简化版：按字符分割（实际应该使用分词器）
    for (size_t i = 0; i < text.length(); ) {
        // 处理UTF-8字符
        unsigned char c = text[i];
        if (c < 0x80) {
            // ASCII字符
            if (c != ' ' && c != '\t' && c != '\n') {
                tokens.push_back(text.substr(i, 1));
            }
            ++i;
        } else if ((c & 0xE0) == 0xC0) {
            // 2字节UTF-8
            tokens.push_back(text.substr(i, 2));
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3字节UTF-8（中文）
            tokens.push_back(text.substr(i, 3));
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4字节UTF-8
            tokens.push_back(text.substr(i, 4));
            i += 4;
        } else {
            ++i;
        }
    }
    return tokens;
}

