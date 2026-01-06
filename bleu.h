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
 * BLEU 分数计算 (BLEU Score Calculation)
 * 
 * 实现机器翻译质量评估的 BLEU 分数计算
 * 
 * BLEU (Bilingual Evaluation Understudy) 是一种基于 n-gram 精确度的评估指标
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

