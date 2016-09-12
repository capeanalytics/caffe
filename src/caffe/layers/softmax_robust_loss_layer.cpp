#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_robust_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  this->ignore_outliers_fraction_ = this->layer_param_.loss_param().ignore_outliers_fraction();
//   printf("INFO: ignoring fraction for outliers is set to [%f]\n", this->ignore_outliers_fraction_);
  
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  
//   printf("INFO: batch size is [%d]\n", outer_num_);  
  this->num_ignored_ = int(this->ignore_outliers_fraction_ * outer_num_);
//   printf("INFO: [%d] examples with highest loss will be ignored\n", this->num_ignored_);
//   exit(1);
  CHECK_GT(this->num_ignored_, 0);
}

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype loss = 0;
  Dtype tmp_loss = 0;
    
  CHECK_EQ(normalization_, LossParameter_NormalizationMode_IGNORE_OUTLIERS)
      << "Normalization has to be set to IGNORE_OUTLIERS for this layer type, instead of: "
      << LossParameter_NormalizationMode_Name(normalization_);
  CHECK_GT(ignore_outliers_fraction_, 0.0);    
  CHECK_LT(ignore_outliers_fraction_, 1.0);  
  
  outliers_.clear();
  
  for (int i = 0; i < outer_num_; ++i) {
    tmp_loss = 0.0;
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      tmp_loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));      
    }
    outliers_.push_back(std::make_pair(tmp_loss, i));      
  }
  std::sort(outliers_.begin(), outliers_.end(), std::greater<std::pair<Dtype, int> >());  
  
//   printf("ignoring [%d] examples with highest loss\n", num_ignored_);
  for (int k = 0; k < outer_num_; ++k) {
    int i = outliers_[k].second;
//     printf("index = [%d]\n", i);
    if (k < num_ignored_) {
//       printf("ignored example at index [%d] with loss [%f]\n", i, outliers_[k].first);
    }
    else {
      loss += outliers_[k].first;
//       printf("added example at index [%d] with loss [%f]\n", i, outliers_[k].first);
    }
  }
  
  int num_examples = outer_num_ - num_ignored_;
  int count = num_examples * inner_num_;
//   printf("robust loss has been computed from [%d] examples: [%f] (at [%d] spatial positions in total)\n", num_examples, loss, count);
  
//   exit(1);
  
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithRobustLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int k = 0; k < outer_num_; ++k) {
      int i = outliers_[k].second;
      if (k < num_ignored_) {
	for (int j = 0; j < inner_num_; ++j) {    
	  for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
	    bottom_diff[i * dim + c * inner_num_ + j] = 0;
	  }
	}
// 	printf("ignored gradient of example [%d]\n", i);
      }
      else {
	for (int j = 0; j < inner_num_; ++j) {    
	  const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	  bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
	  ++count;
	}
// 	printf("added gradient of example [%d]\n", i);
      }
    }
    
//     exit(1);

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithRobustLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithRobustLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithRobustLoss);

}  // namespace caffe
