#include "sublayer_connection.h"

SublayerConnectionImpl::SublayerConnectionImpl(int size, float dropout)
    //: norm(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ size }))),
    : norm(LayerNorm(size)),
    dropout(torch::nn::DropoutOptions(dropout)) {
    register_module("norm", norm);
    register_module("dropout", this->dropout);
}

torch::Tensor SublayerConnectionImpl::forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer) {
    // Layer Norm + 残差连接
    return x + dropout->forward(sublayer(norm->forward(x)));
}

