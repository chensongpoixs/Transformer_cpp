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
 * Beam Search 解码 (Beam Search Decoding)
 * 
 * 实现束搜索算法，用于生成翻译结果
 * 
 * Beam Search 是一种启发式搜索算法，在每一步保留 top-k 个最有可能的候选序列
 * 
 * 算法流程：
 * 1. 初始化 beam，包含 BOS token
 * 2. 对每个时间步：
 *    - 扩展当前 beam 的所有候选
 *    - 计算每个候选的分数
 *    - 保留 top-k 个最佳候选
 * 3. 当遇到 EOS token 时，完成该 beam
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

