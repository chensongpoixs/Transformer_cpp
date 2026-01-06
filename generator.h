#ifndef TRANSFORMER_GENERATOR_H
#define TRANSFORMER_GENERATOR_H

#include <torch/torch.h>

/**
 * 生成器
 * 将解码器输出转换为词汇表大小的概率分布
 */
class GeneratorImpl : public torch::nn::Module {
public:
    GeneratorImpl(int d_model, int vocab_size);
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear proj;  // 线性投影层
};

TORCH_MODULE(Generator);

#endif // TRANSFORMER_GENERATOR_H

