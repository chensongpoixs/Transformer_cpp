#ifndef TRANSFORMER_FEED_FORWARD_H
#define TRANSFORMER_FEED_FORWARD_H

#include <torch/torch.h>

/**
 * 位置前馈神经网络
 * 两层线性变换，中间使用ReLU激活
 */
class PositionwiseFeedForwardImpl : public torch::nn::Module {
public:
    PositionwiseFeedForwardImpl(int d_model, int d_ff, float dropout = 0.1f);
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear w_1;        // 第一个线性层: d_model -> d_ff
    torch::nn::Linear w_2;        // 第二个线性层: d_ff -> d_model
    torch::nn::Dropout dropout;    // Dropout层
};

TORCH_MODULE(PositionwiseFeedForward);

#endif // TRANSFORMER_FEED_FORWARD_H

