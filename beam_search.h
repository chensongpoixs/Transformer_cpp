#ifndef TRANSFORMER_BEAM_SEARCH_H
#define TRANSFORMER_BEAM_SEARCH_H

#include <torch/torch.h>
#include "transformer.h"
#include "utils.h"
#include <vector>
#include <memory>

/**
 * Beam Search解码器
 * 用于生成翻译结果
 */

/**
 * Beam类 - 管理单个beam的状态
 */
class Beam {
public:
    Beam(int size, int pad, int bos, int eos, torch::Device device);
    
    /**
     * 获取当前状态（解码序列）
     */
    torch::Tensor get_current_state();
    
    /**
     * 获取当前来源（backpointer）
     */
    torch::Tensor get_current_origin();
    
    /**
     * 检查是否完成
     */
    bool done() const { return done_; }
    
    /**
     * 推进beam（更新状态并检查是否完成）
     * @param word_logprob 词的对数概率 [beam_size, vocab_size]
     * @return 是否完成
     */
    bool advance(torch::Tensor word_logprob);
    
    /**
     * 排序分数
     */
    std::pair<torch::Tensor, torch::Tensor> sort_scores();
    
    /**
     * 获取最佳分数和索引
     */
    std::pair<float, int> get_the_best_score_and_idx();
    
    /**
     * 获取当前假设（解码序列）
     */
    torch::Tensor get_tentative_hypothesis();
    
    /**
     * 获取完整假设（回溯构建）
     * @param k beam索引
     * @return token ID列表
     */
    std::vector<int> get_hypothesis(int k);

private:
    int size_;
    bool done_;
    int PAD_;
    int BOS_;
    int EOS_;
    torch::Device device_;
    
    torch::Tensor scores_;  // [beam_size]
    std::vector<torch::Tensor> all_scores_;
    std::vector<torch::Tensor> prev_ks_;  // backpointers
    std::vector<torch::Tensor> next_ys_;  // outputs
};

/**
 * Beam Search解码
 * @param model Transformer模型
 * @param src 源语言序列 [batch_size, src_len]
 * @param src_mask 源语言mask [batch_size, 1, src_len]
 * @param max_len 最大解码长度
 * @param pad padding token ID
 * @param bos BOS token ID
 * @param eos EOS token ID
 * @param beam_size beam大小
 * @param device 设备
 * @return 解码结果列表（每个样本一个列表，包含多个候选）和分数列表
 */
std::pair<std::vector<std::vector<std::vector<int>>>, std::vector<std::vector<float>>> 
beam_search(Transformer model,
            torch::Tensor src,
            torch::Tensor src_mask,
            int max_len,
            int pad,
            int bos,
            int eos,
            int beam_size,
            torch::Device device);

#endif // TRANSFORMER_BEAM_SEARCH_H

