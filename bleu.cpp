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

