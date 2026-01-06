#ifndef TRANSFORMER_BLEU_H
#define TRANSFORMER_BLEU_H

#include <string>
#include <vector>
#include <map>

/**
 * BLEU分数计算
 * 实现标准的BLEU-4分数计算
 */

/**
 * 计算单个n-gram的精确度
 * @param candidate 候选句子（token列表）
 * @param reference 参考句子（token列表）
 * @param n n-gram大小（1-4）
 * @return 精确度
 */
float compute_precision(const std::vector<std::string>& candidate,
                        const std::vector<std::string>& reference,
                        int n);

/**
 * 计算修正的n-gram精确度（考虑多个参考句子）
 * @param candidate 候选句子
 * @param references 参考句子列表
 * @param n n-gram大小
 * @return 修正的精确度
 */
float compute_modified_precision(const std::vector<std::vector<std::string>>& references,
                                 const std::vector<std::string>& candidate,
                                 int n);

/**
 * 计算最短参考长度惩罚（brevity penalty）
 * @param candidate_len 候选句子长度
 * @param closest_ref_len 最近的参考句子长度
 * @return 惩罚因子
 */
float brevity_penalty(int candidate_len, int closest_ref_len);

/**
 * 计算BLEU分数
 * @param candidate 候选句子（token列表）
 * @param references 参考句子列表（每个参考是一个token列表）
 * @param max_n 最大n-gram（默认4，即BLEU-4）
 * @return BLEU分数（0-100）
 */
float compute_bleu(const std::vector<std::string>& candidate,
                   const std::vector<std::vector<std::string>>& references,
                   int max_n = 4);

/**
 * 计算语料库级别的BLEU分数
 * @param candidates 候选句子列表
 * @param references 参考句子列表（每个候选对应一个参考列表）
 * @param max_n 最大n-gram（默认4）
 * @return BLEU分数（0-100）
 */
float corpus_bleu(const std::vector<std::vector<std::string>>& candidates,
                  const std::vector<std::vector<std::vector<std::string>>>& references,
                  int max_n = 4);

/**
 * 将字符串按空格分割为token列表
 */
std::vector<std::string> tokenize(const std::string& text);

/**
 * 中文分词（简化版，按字符分割）
 */
std::vector<std::string> tokenize_chinese(const std::string& text);

#endif // TRANSFORMER_BLEU_H

