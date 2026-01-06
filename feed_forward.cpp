#include "feed_forward.h"

PositionwiseFeedForwardImpl::PositionwiseFeedForwardImpl(int d_model, int d_ff, float drop_rate)
    : w_1(torch::nn::LinearOptions(d_model, d_ff)),
      w_2(torch::nn::LinearOptions(d_ff, d_model)),
      dropout(torch::nn::DropoutOptions(drop_rate)) {
    register_module("w_1", w_1);
    register_module("w_2", w_2);
    register_module("dropout", dropout);
}

torch::Tensor PositionwiseFeedForwardImpl::forward(torch::Tensor x) {
    // FFN(x) = ReLU(W1 * x + b1) -> Dropout -> W2 * x + b2
    return w_2->forward(dropout->forward(torch::relu(w_1->forward(x))));
}

