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
 * 层归一化实现 (Layer Normalization Implementation)
 * 
 * 实现层归一化的前向传播
 * 
 * 计算特征维度的均值和方差，然后进行归一化和缩放
				   
				   
				   
				   
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

#include "layer_norm.h"

LayerNormImpl::LayerNormImpl(int features, float eps)
    : eps(eps) {
    // 初始化α为全1, 而β为全0，并注册为可训练参数
    // 使用register_parameter来注册参数，它会返回一个torch::Tensor引用
    a_2 = register_parameter("a_2", torch::ones(features));
    b_2 = register_parameter("b_2", torch::zeros(features));
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
    // 计算均值和标准差（keepdim=True保持维度）
    // x的形状: [batch_size, seq_len, d_model] 例如 [2, 10, 512]
    // 在最后一个维度上计算均值和标准差
    // Python版本: mean = x.mean(-1, keepdim=True), std = x.std(-1, keepdim=True)
    
    // 计算均值和标准差（keepdim=True保持维度）
    // 使用-1表示最后一个维度，与Python版本一致
    auto mean = x.mean(-1, true);  // [batch_size, seq_len, 1] 例如 [2, 10, 1]
    
    // 手动计算方差和标准差，确保形状正确
    // 方差 = mean((x - mean)^2)，使用有偏估计
    auto x_centered = x - mean;  // [batch_size, seq_len, d_model] - [batch_size, seq_len, 1] -> [batch_size, seq_len, d_model]
    auto var = (x_centered * x_centered).mean(-1, true);  // [batch_size, seq_len, 1]
    auto std = torch::sqrt(var + eps);  // [batch_size, seq_len, 1]
    
    // Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
    // 其中a和b是可学习的参数，eps是为了防止除以0的小常数
    // Python版本: (x - mean) / torch.sqrt(std ** 2 + self.eps)
    
    // 计算归一化部分: (x - mean) / sqrt(std^2 + eps)
    // x的形状: [batch, seq_len, d_model] = [2, 10, 512]
    // mean的形状: [batch, seq_len, 1] = [2, 10, 1]
    // std的形状: [batch, seq_len, 1] = [2, 10, 1]
    // (x - mean) 应该得到 [2, 10, 512] (广播)
    // std * std + eps 应该得到 [2, 10, 1]
    // 除法应该得到 [2, 10, 512]
    auto normalized = (x - mean) / torch::sqrt(std * std + eps);

    // 应用缩放和偏移，a_2和b_2会自动广播
    // 确保a_2和b_2在正确的设备上
    auto a_2_device = a_2.to(x.device());
    auto b_2_device = b_2.to(x.device());

    return a_2_device * normalized + b_2_device;
    

    
}



#if 0


#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// LayerNorm 类
class LayerNorm {
public:
    // 构造函数，接收输入维度
    LayerNorm(int dim) : dim_(dim), beta_(dim, 0.0), gamma_(dim, 1.0) {}

    // 前向传播
    vector<double> forward(const vector<double>& input) {
        // 检查输入维度
        if (input.size() != dim_) {
            throw invalid_argument("Input dimension mismatch.");
        }

        // 1. 计算均值
        double sum = 0.0;
        for (double x : input) {
            sum += x;
        }
        double mean = sum / dim_;

        // 2. 计算方差
        double sq_sum = 0.0;
        for (double x : input) {
            sq_sum += pow(x - mean, 2);
        }
        double variance = sq_sum / dim_;

        // 3. 计算标准差
        double stddev = sqrt(variance);

        // 4. 标准化
        vector<double> normalized(dim_);
        for (int i = 0; i < dim_; ++i) {
            normalized[i] = (input[i] - mean) / stddev;
        }

        // 5. 应用缩放和偏移
        vector<double> output(dim_);
        for (int i = 0; i < dim_; ++i) {
            output[i] = gamma_[i] * normalized[i] + beta_[i];
        }

        // 更新统计信息 (用于训练或在线计算)
        mean_ = mean;
        stddev_ = stddev;

        return output;
    }

    // 设置学习率，用于更新 beta 和 gamma
    void set_learning_rate(double lr) {
        learning_rate_ = lr;
    }

    // 反向传播 (简化版本，仅用于演示目的)
    //  在实际应用中，需要考虑梯度计算和更新 beta 和 gamma
    void backward(const vector<double>& grad_output) {
        // 简化版本，仅更新 beta 和 gamma
        if (grad_output.size() != dim_) {
            throw invalid_argument("Gradient output dimension mismatch.");
        }

        for (int i = 0; i < dim_; ++i) {
            // 假设输入是标准化的
            beta_[i] -= learning_rate_ * grad_output[i];
            gamma_[i] -= learning_rate_ * grad_output[i] * (input_[i] - mean_) / stddev_;  // 假设 input_ 存储了前向传播的输入
        }
    }

    // 设置参数 (用于初始化或加载参数)
    void set_params(const vector<double>& beta, const vector<double>& gamma) {
        if (beta.size() != dim_ || gamma.size() != dim_) {
            throw invalid_argument("Parameter dimensions mismatch.");
        }
        beta_ = beta;
        gamma_ = gamma;
    }

    // 获取 beta 参数
    vector<double> get_beta() const {
        return beta_;
    }

    // 获取 gamma 参数
    vector<double> get_gamma() const {
        return gamma_;
    }
private:
    int dim_;
    vector<double> beta_;
    vector<double> gamma_;
    double mean_ = 0.0;  // 用于存储均值 (可选)
    double stddev_ = 0.0; // 用于存储标准差 (可选)
    double learning_rate_ = 0.01; // 学习率，用于更新 beta 和 gamma
    vector<double> input_;  // 保存前向传播的输入，用于反向传播
};


int main() {
    // 示例用法
    LayerNorm layerNorm(4);

    // 输入数据
    vector<double> input = { 1.0, 2.0, 3.0, 4.0 };

    // 前向传播
    vector<double> output = layerNorm.forward(input);

    cout << "Input: ";
    for (double x : input) {
        cout << x << " ";
    }
    cout << endl;

    cout << "Output: ";
    for (double x : output) {
        cout << x << " ";
    }
    cout << endl;

    // 设置学习率
    layerNorm.set_learning_rate(0.01);

    // 假设我们有一个反向传播的梯度输出
    vector<double> grad_output = { 0.1, -0.2, 0.3, -0.4 };

    // 反向传播 (简化版本)
    layerNorm.backward(grad_output);
    layerNorm.input_ = input; // 存储输入用于反向传播

    cout << "Beta after backward pass: ";
    vector<double> beta = layerNorm.get_beta();
    for (double x : beta) {
        cout << x << " ";
    }
    cout << endl;

    cout << "Gamma after backward pass: ";
    vector<double> gamma = layerNorm.get_gamma();
    for (double x : gamma) {
        cout << x << " ";
    }
    cout << endl;

    return 0;
}

#endif // 