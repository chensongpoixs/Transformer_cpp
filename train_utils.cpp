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
 * 训练工具类实现 (Training Utilities Implementation)
 * 
 * 实现 NoamOpt 优化器和 LossCompute 损失计算类
 * 
 * NoamOpt: 实现学习率预热和衰减策略
 * LossCompute: 计算交叉熵损失并执行反向传播
				   
				   
				   
				   
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

#include "train_utils.h"
#include <cmath>

// NoamOpt实现
NoamOpt::NoamOpt(int model_size, float factor, int warmup, 
                 std::shared_ptr<torch::optim::Optimizer> optimizer)
    : model_size(model_size), factor(factor), warmup(warmup), 
      optimizer(optimizer), current_step(0), current_rate(0.0f) {
}

void NoamOpt::step() {
    current_step++;
    float lr = rate();
    current_rate = lr;
    
    // 更新优化器的学习率
    for (auto& param_group : optimizer->param_groups()) {
        param_group.options().set_lr(lr);
    }
    
    optimizer->step();
    optimizer->zero_grad();
}

float NoamOpt::rate(int step) {
    if (step < 0) {
        step = current_step;
    }
    
    // 学习率计算公式：factor * (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
    float lr = factor * std::pow(static_cast<float>(model_size), -0.5f) * 
               std::min(std::pow(static_cast<float>(step), -0.5f), 
                       static_cast<float>(step) * std::pow(static_cast<float>(warmup), -1.5f));
    return lr;
}

// get_std_opt实现
std::shared_ptr<NoamOpt> get_std_opt(Transformer model, int d_model) {
    // 创建Adam优化器
    auto adam_optimizer = std::make_shared<torch::optim::Adam>(
        model->parameters(),
        torch::optim::AdamOptions(0.0).betas(std::make_tuple(0.9, 0.98)).eps(1e-9)
    );
    
    // 创建NoamOpt优化器
    return std::make_shared<NoamOpt>(d_model, 1.0f, 10000, adam_optimizer);
}

// LossCompute实现
LossCompute::LossCompute(Generator generator, 
                         torch::nn::CrossEntropyLoss criterion,
                         std::shared_ptr<NoamOpt> opt)
    : generator(generator), criterion(criterion), opt(opt) {
}

float LossCompute::operator()(torch::Tensor out, torch::Tensor targets, float normalize) {
    // 原有实现：立即提取 loss 值（保持向后兼容）
    auto [loss_tensor, has_backward] = compute_loss_tensor(out, targets, normalize);
    
    // 提取损失值（强制同步）
    float loss_value = loss_tensor.item<float>();
    
    // 显式释放 loss tensor
    loss_tensor = torch::Tensor();
    
    return loss_value;
}

std::pair<torch::Tensor, bool> LossCompute::compute_loss_tensor(torch::Tensor out, 
                                                                 torch::Tensor targets, 
                                                                 float normalize) {
    // out: [batch_size, seq_len, vocab_size]
    // targets: [batch_size, seq_len]
    
    // 通过generator得到log概率
    auto log_probs = generator->forward(out);  // [batch_size, seq_len, vocab_size]
    
    // 重塑为2D: [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
    auto log_probs_flat = log_probs.view({-1, log_probs.size(-1)});
    auto targets_flat = targets.contiguous().view(-1);
    
    // 计算损失（返回未归一化的 loss tensor）
    auto loss = criterion(log_probs_flat, targets_flat);
    
    // 如果提供了优化器，进行反向传播
    bool has_backward = false;
    if (opt != nullptr) {
        loss.backward();
        opt->step();  // NoamOpt的step()已经包含了optimizer->step()和zero_grad()
        has_backward = true;
    }
    
    // 归一化 loss（但保持为 tensor，不提取值）
    auto normalized_loss = loss / normalize;
    
    // 显式释放中间张量（帮助释放显存）
    // 注意：不释放 loss，因为需要返回
    log_probs_flat = torch::Tensor();
    targets_flat = torch::Tensor();
    log_probs = torch::Tensor();
    
    // 返回归一化后的 loss tensor 和是否执行了反向传播
    return {normalized_loss, has_backward};
}

