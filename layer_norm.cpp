#include "layer_norm.h"
//
//LayerNormImpl::LayerNormImpl(int features, float eps)
//    : eps(eps) {
//    // 初始化α为全1, 而β为全0，并注册为可训练参数
//    // 使用register_parameter来注册参数，它会返回一个torch::Tensor引用
//    a_2 = register_parameter("a_2", torch::ones(features));
//    b_2 = register_parameter("b_2", torch::zeros(features));
//}
//
//torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
//    // 计算均值和标准差（keepdim=True保持维度）
//    // x的形状: [batch_size, seq_len, d_model] 例如 [2, 10, 512]
//    // 在最后一个维度上计算均值和标准差
//    // Python版本: mean = x.mean(-1, keepdim=True), std = x.std(-1, keepdim=True)
//    
//    // 使用维度列表，-1表示最后一个维度
//    std::vector<int64_t> dims = {-1};
//    
//    // 计算均值，keepdim=true保持维度
//    auto mean = x.mean(dims, true);  // [batch_size, seq_len, 1] 例如 [2, 10, 1]
//    
//    // 计算标准差（使用有偏估计，与PyTorch的LayerNorm一致）
//    // unbiased=false表示有偏估计（除以n而不是n-1）
//    // Python版本: std = x.std(-1, keepdim=True) 默认使用有偏估计
//    auto std = x.std(dims, true, false);  // [batch_size, seq_len, 1]
//    
//    // Layer Norm公式: y = a * (x - mean) / sqrt(std^2 + eps) + b
//    // 其中a和b是可学习的参数，eps是为了防止除以0的小常数
//    // Python版本: (x - mean) / torch.sqrt(std ** 2 + self.eps)
//    
//    // 确保a_2和b_2在正确的设备上
//    auto a_2_device = a_2.to(x.device());
//    auto b_2_device = b_2.to(x.device());
//    
//    // 计算归一化部分: (x - mean) / sqrt(std^2 + eps)
//    // x - mean: [batch, seq_len, d_model] - [batch, seq_len, 1] -> [batch, seq_len, d_model] (广播)
//    // std: [batch, seq_len, 1]，需要先平方再加eps再开方
//    auto normalized = (x - mean) / torch::sqrt(std * std + eps);
//    
//    // 应用缩放和偏移: a_2 * normalized + b_2
//    // normalized的形状: [batch, seq_len, d_model] 例如 [2, 10, 512]
//    // a_2和b_2的形状: [d_model] 例如 [512]
//    // 为了确保正确广播，我们需要将a_2和b_2 reshape为可以广播的形状
//    // 对于3D输入 [batch, seq_len, d_model]，我们需要 [1, 1, d_model]
//    
//    // 使用unsqueeze添加前导维度，使其能够正确广播
//    // a_2: [d_model] -> unsqueeze(0) -> [1, d_model] -> unsqueeze(0) -> [1, 1, d_model]
//    int ndim = x.dim();
//    auto a_2_reshaped = a_2_device;
//    auto b_2_reshaped = b_2_device;
//    
//    // 添加前导维度直到维度数匹配
//    for (int i = 0; i < ndim - 1; ++i) {
//        a_2_reshaped = a_2_reshaped.unsqueeze(0);
//        b_2_reshaped = b_2_reshaped.unsqueeze(0);
//    }
//    
//    // 现在 a_2_reshaped 和 b_2_reshaped 的形状是 [1, 1, d_model] (对于3D输入)
//    // 可以正确广播到 [batch, seq_len, d_model]
//    return a_2_reshaped * normalized + b_2_reshaped;
//}



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