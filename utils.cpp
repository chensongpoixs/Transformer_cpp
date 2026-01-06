#include "utils.h"
#include <cmath>

torch::Tensor subsequent_mask(int size) {
    // 创建上三角矩阵（右上角为1，左下角为0）
    auto mask = torch::triu(torch::ones({1, size, size}), 1);
    // 返回布尔掩码（右上角为False，左下角为True）
    return mask == 0;
}

void xavier_uniform_init(torch::Tensor& tensor) {
    if (tensor.dim() > 1) {
        float gain = std::sqrt(2.0f / (tensor.size(0) + tensor.size(1)));
        float bound = std::sqrt(3.0f) * gain;
        torch::nn::init::uniform_(tensor, -bound, bound);
    }
}

