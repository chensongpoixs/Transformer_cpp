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
 * 训练相关函数声明 (Training Functions Declaration)
 * 
 * 包含训练、验证和评估相关的函数声明：
 * - run_epoch: 运行一个 epoch 的训练或验证
 * - train: 主训练函数，包含完整的训练循环
 * - evaluate: 评估函数，计算 BLEU 分数
				   
				   
				   
				   
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

#ifndef TRANSFORMER_TRAIN_H
#define TRANSFORMER_TRAIN_H

#include <torch/torch.h>
#include "transformer.h"
#include "train_utils.h"
#include "data_loader.h"
#include "config.h"
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <tuple>

/**
 * 运行一个epoch的训练或验证
 * @param dataset 数据集
 * @param model Transformer模型
 * @param loss_compute 损失计算器
 * @param batch_size 批次大小
 * @param device 设备
 * @param config 配置
 * @return (平均损失, 总tokens数, 批次数量)
 */
std::tuple<float, long long, size_t> run_epoch(MTDataset& dataset,
                                                Transformer model,
                                                LossCompute& loss_compute,
                                                int batch_size,
                                                torch::Device device,
                                                const TransformerConfig& config,
                                                bool is_training = true,
                                                int epoch = 0,
                                                int total_epochs = 0);

/**
 * 训练函数
 * @param train_dataset 训练数据集
 * @param dev_dataset 验证数据集
 * @param model Transformer模型
 * @param criterion 损失函数
 * @param optimizer 优化器
 * @param config 配置
 * @param device 设备
 */
void train(MTDataset& train_dataset,
           MTDataset& dev_dataset,
           Transformer model,
           torch::nn::CrossEntropyLoss criterion,
           std::shared_ptr<NoamOpt> optimizer,
           const TransformerConfig& config,
           torch::Device device);

/**
 * 评估函数（计算BLEU分数）
 * 注意：完整实现需要集成SentencePiece和BLEU计算库
 * @param dataset 数据集
 * @param model Transformer模型
 * @param config 配置
 * @param device 设备
 * @return BLEU分数
 */
float evaluate(MTDataset& dataset,
                Transformer model,
                const TransformerConfig& config,
                torch::Device device);

#endif // TRANSFORMER_TRAIN_H

