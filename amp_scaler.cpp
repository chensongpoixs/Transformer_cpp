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
 * 混合精度训练缩放器实现 (AMP Scaler Implementation)
				   
				   
				   
				   
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

#include "amp_scaler.h"
#include <algorithm>
#include <cmath>

AMPScaler::AMPScaler(float init_scale, int scale_window)
    : scale_(init_scale), init_scale_(init_scale), scale_window_(scale_window),
      step_count_(0), found_inf_(false) {
}

torch::Tensor AMPScaler::scale(torch::Tensor loss) {
    // 缩放 loss，防止 FP16 梯度下溢
    return loss * scale_;
}

void AMPScaler::unscale(std::shared_ptr<torch::optim::Optimizer> optimizer) {
    // 检查所有参数的梯度是否包含 inf/nan
    found_inf_ = false;
    
    // 遍历优化器的所有参数组
    for (auto& param_group : optimizer->param_groups()) {
        for (auto& param : param_group.params()) {
            if (param.grad().defined()) {
                if (check_inf_nan(param.grad())) {
                    found_inf_ = true;
                    break;
                }
            }
        }
        if (found_inf_) {
            break;
        }
    }
    
    // 如果没有溢出，缩放梯度
    if (!found_inf_) {
        for (auto& param_group : optimizer->param_groups()) {
            for (auto& param : param_group.params()) {
                if (param.grad().defined()) {
                    param.mutable_grad().div_(scale_);
                }
            }
        }
    }
}

void AMPScaler::update() {
    step_count_++;
    
    // 每 scale_window_ 次迭代更新一次缩放因子
    if (step_count_ % scale_window_ == 0) {
        if (found_inf_) {
            // 发现溢出，减小缩放因子
            scale_ = std::max(scale_ / 2.0f, 1.0f);
        } else {
            // 没有溢出，可以尝试增大缩放因子
            scale_ = std::min(scale_ * 2.0f, init_scale_ * 2.0f);
        }
    }
    
    found_inf_ = false;  // 重置溢出标志
}

bool AMPScaler::check_inf_nan(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) {
        return false;
    }
    
    // 检查是否包含 inf 或 nan
    auto inf_mask = torch::isinf(tensor);
    auto nan_mask = torch::isnan(tensor);
    
    // 如果有任何 inf 或 nan，返回 true
    if (inf_mask.any().item<bool>() || nan_mask.any().item<bool>()) {
        return true;
    }
    
    return false;
}

