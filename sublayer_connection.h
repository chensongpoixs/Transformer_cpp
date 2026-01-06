#ifndef TRANSFORMER_SUBLAYER_CONNECTION_H
#define TRANSFORMER_SUBLAYER_CONNECTION_H

#include <torch/torch.h>
#include "layer_norm.h"
#include <functional>

/**
 * 子层连接
 * 将Multi-Head Attention和Feed Forward层连接在一起
 * 每一层输出之后都要先做Layer Norm再残差连接
 */
class SublayerConnectionImpl : public torch::nn::Module {
public:
    SublayerConnectionImpl(int size, float dropout);
    
    // 使用函数指针或std::function来处理sublayer
    torch::Tensor forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer);

private:
    LayerNorm norm;
   // torch::nn::LayerNorm norm;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(SublayerConnection);

#endif // TRANSFORMER_SUBLAYER_CONNECTION_H

