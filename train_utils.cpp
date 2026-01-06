#include "train_utils.h"
#include <cmath>

// NoamOpt实现
NoamOpt::NoamOpt(int model_size, float factor, int warmup, 
                 std::shared_ptr<torch::optim::Optimizer> optimizer)
    : model_size(model_size), factor(factor), warmup(warmup), 
      optimizer(optimizer), current_step(0), current_rate(0.0f) {
}

void NoamOpt::step() {
    current_step++;
    float lr = rate();
    current_rate = lr;
    
    // 更新优化器的学习率
    for (auto& param_group : optimizer->param_groups()) {
        param_group.options().set_lr(lr);
    }
    
    optimizer->step();
    optimizer->zero_grad();
}

float NoamOpt::rate(int step) {
    if (step < 0) {
        step = current_step;
    }
    
    // 学习率计算公式：factor * (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)
    float lr = factor * std::pow(static_cast<float>(model_size), -0.5f) * 
               std::min(std::pow(static_cast<float>(step), -0.5f), 
                       static_cast<float>(step) * std::pow(static_cast<float>(warmup), -1.5f));
    return lr;
}

// get_std_opt实现
std::shared_ptr<NoamOpt> get_std_opt(Transformer model, int d_model) {
    // 创建Adam优化器
    auto adam_optimizer = std::make_shared<torch::optim::Adam>(
        model->parameters(),
        torch::optim::AdamOptions(0.0).betas(std::make_tuple(0.9, 0.98)).eps(1e-9)
    );
    
    // 创建NoamOpt优化器
    return std::make_shared<NoamOpt>(d_model, 1.0f, 10000, adam_optimizer);
}

// LossCompute实现
LossCompute::LossCompute(Generator generator, 
                         torch::nn::CrossEntropyLoss criterion,
                         std::shared_ptr<NoamOpt> opt)
    : generator(generator), criterion(criterion), opt(opt) {
}

float LossCompute::operator()(torch::Tensor out, torch::Tensor targets, float normalize) {
    // out: [batch_size, seq_len, vocab_size]
    // targets: [batch_size, seq_len]
    
    // 通过generator得到log概率
    auto log_probs = generator->forward(out);  // [batch_size, seq_len, vocab_size]
    
    // 重塑为2D: [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
    auto log_probs_flat = log_probs.view({-1, log_probs.size(-1)});
    auto targets_flat = targets.contiguous().view(-1);
    
    // 计算损失
    auto loss = criterion(log_probs_flat, targets_flat);
    
    // 如果提供了优化器，进行反向传播
    if (opt != nullptr) {
        loss.backward();
        opt->step();  // NoamOpt的step()已经包含了optimizer->step()和zero_grad()
    }
    
    // 返回归一化后的损失
    return loss.item<float>() / normalize;
}

