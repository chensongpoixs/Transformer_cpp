#ifndef TRANSFORMER_LAYER_NORM_H
#define TRANSFORMER_LAYER_NORM_H

#include <torch/torch.h>

/**
 * 层归一化实现
 */
class LayerNormImpl : public torch::nn::Module {
public:
    LayerNormImpl(int features, float eps = 1e-6f);
    //前向传播函数
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor a_2;  // 缩放参数（初始化为1）
    torch::Tensor b_2;  // 偏移参数（初始化为0）
    float eps;          // 平滑项
};





TORCH_MODULE(LayerNorm);



//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <algorithm>
//
//using namespace std;
/*

代码解释:

LayerNorm 类:

dim_: 输入向量的维度。
beta_: 缩放参数，每个维度一个。初始化为 0。
gamma_: 偏移参数，每个维度一个。初始化为 1。
mean_, stddev_: 用于存储均值和标准差，在某些情况下可能需要。
learning_rate_: 学习率，用于更新 beta_ 和 gamma_。
input_: 用于保存前向传播的输入，以便在反向传播中使用。
LayerNorm(int dim) 构造函数:

初始化 dim_ 为输入维度。
初始化 beta_ 和 gamma_ 为指定维度的向量，所有元素初始化为 0 和 1。
forward(const vector<double>& input) 方法:

输入验证: 检查输入向量的维度是否与 dim_ 匹配。
计算均值: 计算输入向量的均值。
计算方差: 计算输入向量的方差。
计算标准差: 计算输入向量的标准差。
标准化: 将输入向量标准化，使其均值为 0，标准差为 1。
缩放和偏移: 将标准化后的向量缩放（乘以 gamma_）并偏移（加上 beta_）。
更新统计信息: 保存计算出的均值和标准差，以便在训练过程中使用。 这部分可以根据具体需求进行调整。
返回输出: 返回缩放和偏移后的向量。
set_learning_rate(double lr) 方法:

设置学习率，用于更新 beta_ 和 gamma_。
backward(const vector<double>& grad_output) 方法:

输入验证: 检查梯度输出的维度是否与 dim_ 匹配。
参数更新: 简化版本的反向传播，仅更新 beta_ 和 gamma_。 更完整的实现需要计算梯度并更新权重。 这个例子假设输入已经标准化，所以更新规则会包含标准化因子。
保存输入: 将前向传播的输入保存到 input_，以便在反向传播中使用。
set_params(const vector<double>& beta, const vector<double>& gamma) 方法:

设置 beta_ 和 gamma_ 的值。
get_beta() 和 get_gamma() 方法:

获取 beta_ 和 gamma_ 的值。
重要说明:

反向传播: 提供的 backward() 方法是一个简化的版本。 在实际的深度学习框架中，反向传播需要计算更复杂的梯度，并使用优化算法来更新 beta_ 和 gamma_。
均值和标准差的估计: 在训练过程中，通常使用批处理数据来估计均值和标准差。 这可以提高训练的效率和稳定性。 可以使用滑动平均来估计均值和标准差。
实现细节: 不同的深度学习框架可能对 LayerNorm 的实现细节有所不同，例如计算均值和方差的方式、参数更新的策略等。
CUDA/GPU加速: 在实际应用中，LayerNorm 的计算通常在 GPU 上进行加速，以提高性能。 这需要使用 CUDA 或其他 GPU 编程技术。
如何改进这个实现:

使用滑动平均: 使用滑动平均来估计均值和标准差，以提高训练的稳定性和效率。
实现完整的反向传播: 计算 beta_ 和 gamma_ 的梯度，并使用优化算法来更新它们。
添加 CUDA/GPU 加速: 使用 CUDA 或其他 GPU 编程技术，将 LayerNorm 的计算放在 GPU 上进行加速。
支持不同的数据类型: 支持不同的数据类型，例如 float 和 double。
添加单元测试: 编写单元测试来验证 LayerNorm 的正确性。
支持批量数据: 修改代码以处理批量数据，而不是单个输入向量。 这通常需要计算整个批次的均值和方差。
考虑稳定性: 在计算方差时，可以添加一个小的 epsilon 值，以避免除以零的情况。
实现不同的 LayerNorm 变体: 例如，GroupNorm 或 InstanceNorm。

*/
// LayerNorm 类
//class LayerNorm {
//public:
//    // 构造函数，接收输入维度
//    LayerNorm(int dim) : dim_(dim), beta_(dim, 0.0), gamma_(dim, 1.0) {}
//
//    // 前向传播
//    vector<double> forward(const vector<double>& input) {
//        // 检查输入维度
//        if (input.size() != dim_) {
//            throw invalid_argument("Input dimension mismatch.");
//        }
//
//        // 1. 计算均值
//        double sum = 0.0;
//        for (double x : input) {
//            sum += x;
//        }
//        double mean = sum / dim_;
//
//        // 2. 计算方差
//        double sq_sum = 0.0;
//        for (double x : input) {
//            sq_sum += pow(x - mean, 2);
//        }
//        double variance = sq_sum / dim_;
//
//        // 3. 计算标准差
//        double stddev = sqrt(variance);
//
//        // 4. 标准化
//        vector<double> normalized(dim_);
//        for (int i = 0; i < dim_; ++i) {
//            normalized[i] = (input[i] - mean) / stddev;
//        }
//
//        // 5. 应用缩放和偏移
//        vector<double> output(dim_);
//        for (int i = 0; i < dim_; ++i) {
//            output[i] = gamma_[i] * normalized[i] + beta_[i];
//        }
//
//        // 更新统计信息 (用于训练或在线计算)
//        mean_ = mean;
//        stddev_ = stddev;
//
//        return output;
//    }
//
//    // 设置学习率，用于更新 beta 和 gamma
//    void set_learning_rate(double lr) {
//        learning_rate_ = lr;
//    }
//
//    // 反向传播 (简化版本，仅用于演示目的)
//    //  在实际应用中，需要考虑梯度计算和更新 beta 和 gamma
//    void backward(const vector<double>& grad_output) {
//        // 简化版本，仅更新 beta 和 gamma
//        if (grad_output.size() != dim_) {
//            throw invalid_argument("Gradient output dimension mismatch.");
//        }
//
//        for (int i = 0; i < dim_; ++i) {
//            // 假设输入是标准化的
//            beta_[i] -= learning_rate_ * grad_output[i];
//            gamma_[i] -= learning_rate_ * grad_output[i] * (input_[i] - mean_) / stddev_;  // 假设 input_ 存储了前向传播的输入
//        }
//    }
//
//    // 设置参数 (用于初始化或加载参数)
//    void set_params(const vector<double>& beta, const vector<double>& gamma) {
//        if (beta.size() != dim_ || gamma.size() != dim_) {
//            throw invalid_argument("Parameter dimensions mismatch.");
//        }
//        beta_ = beta;
//        gamma_ = gamma;
//    }
//
//    // 获取 beta 参数
//    vector<double> get_beta() const {
//        return beta_;
//    }
//
//    // 获取 gamma 参数
//    vector<double> get_gamma() const {
//        return gamma_;
//    }
//public:
//    int dim_;
//    vector<double> beta_;
//    vector<double> gamma_;
//    double mean_ = 0.0;  // 用于存储均值 (可选)
//    double stddev_ = 0.0; // 用于存储标准差 (可选)
//    double learning_rate_ = 0.01; // 学习率，用于更新 beta 和 gamma
//    vector<double> input_;  // 保存前向传播的输入，用于反向传播
//};

#if 0

int test_layer_main() {
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

#endif // TRANSFORMER_LAYER_NORM_H

