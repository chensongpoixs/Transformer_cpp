#ifndef TRANSFORMER_TRAIN_UTILS_H
#define TRANSFORMER_TRAIN_UTILS_H

#include <torch/torch.h>
#include "transformer.h"
#include <memory>

/**
 * Noam优化器
 * 实现学习率预热和衰减策略
 */
class NoamOpt {
public:
    NoamOpt(int model_size, float factor, int warmup, 
            std::shared_ptr<torch::optim::Optimizer> optimizer);
    
    void step();
    float rate(int step = -1);

private:
    int model_size;
    float factor;
    int warmup;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
    int current_step;
    float current_rate;
};

/**
 * 创建标准的Noam优化器
 * @param model Transformer模型
 * @param d_model 模型维度（从配置传入）
 * @return NoamOpt优化器
 */
std::shared_ptr<NoamOpt> get_std_opt(Transformer model, int d_model);

/**
 * 损失计算类（简化版，单GPU）
 */
class LossCompute {
public:
    LossCompute(Generator generator, 
                torch::nn::CrossEntropyLoss criterion,
                std::shared_ptr<NoamOpt> opt = nullptr);
    
    float operator()(torch::Tensor out, torch::Tensor targets, float normalize);

private:
    Generator generator;
    torch::nn::CrossEntropyLoss criterion;
    std::shared_ptr<NoamOpt> opt;
};

#endif // TRANSFORMER_TRAIN_UTILS_H

