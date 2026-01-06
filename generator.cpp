#include "generator.h"

GeneratorImpl::GeneratorImpl(int d_model, int vocab_size)
    : proj(torch::nn::LinearOptions(d_model, vocab_size)) {
    register_module("proj", proj);
}

torch::Tensor GeneratorImpl::forward(torch::Tensor x) {
    // 线性投影 + log_softmax
    return torch::log_softmax(proj->forward(x), -1);
}

