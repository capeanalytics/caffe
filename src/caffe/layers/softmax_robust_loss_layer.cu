#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_robust_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << this->type()
               << " Layer not implemented yet for GPU.";  
}

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << this->type()
               << " Layer not implemented yet for GPU.";
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithRobustLossLayer);

}  // namespace caffe
