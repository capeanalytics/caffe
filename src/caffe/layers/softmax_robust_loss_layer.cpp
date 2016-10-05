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
  
  method_ = 0;
  if (this->layer_param_.loss_param().has_ignore_outliers_fraction()) {
    CHECK_EQ(this->layer_param_.loss_param().has_stdev_z_value(), false);
    CHECK_EQ(this->layer_param_.loss_param().has_retained_inlier_fraction(), false);
    ignore_outliers_fraction_ = this->layer_param_.loss_param().ignore_outliers_fraction();
    // printf("INFO: ignoring fraction for outliers is set to [%f]\n", this->ignore_outliers_fraction_);
    CHECK_GT(ignore_outliers_fraction_, 0.0);    
    CHECK_LT(ignore_outliers_fraction_, 1.0);  
    method_ = 1;
  }
  if (this->layer_param_.loss_param().has_stdev_z_value()) {
    CHECK_EQ(this->layer_param_.loss_param().has_ignore_outliers_fraction(), false);
    CHECK_EQ(this->layer_param_.loss_param().has_retained_inlier_fraction(), false);    
    stdev_z_value_ = this->layer_param_.loss_param().stdev_z_value();
    CHECK_GT(stdev_z_value_, 0.0);    
    method_ = 2;
    CHECK_EQ(this->layer_param_.loss_param().has_loss_cache_size(), true);
    loss_cache_size_ = this->layer_param_.loss_param().loss_cache_size();
  }
  if (this->layer_param_.loss_param().has_retained_inlier_fraction()) {
    CHECK_EQ(this->layer_param_.loss_param().has_ignore_outliers_fraction(), false);
    CHECK_EQ(this->layer_param_.loss_param().has_stdev_z_value(), false);       
    retained_inlier_fraction_ = this->layer_param_.loss_param().retained_inlier_fraction();
    CHECK_GT(retained_inlier_fraction_, 0.0);
    CHECK_LT(retained_inlier_fraction_, 1.0);
    method_ = 3;
    batch_count_ = 0;
    // TODO: magic numbers!
    int min_iteration_count = 5000;
    int batches_per_iteration = 4;
    min_batch_count_ = min_iteration_count * batches_per_iteration;
  }
  CHECK_GT(method_, 0);
  normalization_ = this->layer_param_.loss_param().normalization();
  CHECK_EQ(normalization_, LossParameter_NormalizationMode_IGNORE_OUTLIERS)
      << "Normalization has to be set to IGNORE_OUTLIERS for this layer type, instead of: "
      << LossParameter_NormalizationMode_Name(normalization_);  
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
  CHECK_EQ(inner_num_, 1)
      << "Not yet verified whether this implementation makes any sense for output volumes with "
      << "a spatial extent!";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
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
  Dtype tmp_residual = 0;
  int count = 0;
  
  if (method_ == 1) {
    // printf("INFO: batch size is [%d]\n", outer_num_);  
    num_ignored_ = int(ignore_outliers_fraction_ * outer_num_);
    // printf("INFO: [%d] examples with highest loss will be ignored\n", this->num_ignored_);
    CHECK_GT(num_ignored_, 0);    
    outliers_.clear();
    
    for (int i = 0; i < outer_num_; ++i) {
      tmp_loss = 0.0;
      for (int j = 0; j < inner_num_; j++) {
	const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	DCHECK_GE(label_value, 0);
	DCHECK_LT(label_value, prob_.shape(softmax_axis_));
	tmp_loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)));      
      }
      outliers_.push_back(std::make_pair(tmp_loss, i));      
    }
    std::sort(outliers_.begin(), outliers_.end(), std::greater<std::pair<Dtype, int> >());  
    
    // printf("ignoring [%d] examples with highest loss\n", num_ignored_);
    for (int k = 0; k < outer_num_; ++k) {
      // int i = outliers_[k].second;
      // printf("index = [%d]\n", i);
      if (k < num_ignored_) {
        // printf("ignored example at index [%d] with loss [%f]\n", i, outliers_[k].first);
      }
      else {
	loss += outliers_[k].first;
        // printf("added example at index [%d] with loss [%f]\n", i, outliers_[k].first);
      }
    }
    
    int num_examples = outer_num_ - num_ignored_;
    count = num_examples * inner_num_;
    // printf("robust loss has been computed from [%d] examples: [%f] (at [%d] spatial positions in total)\n", num_examples, loss, count);
  }
  if (method_ == 2) {
    sorted_cache_.clear();
    is_outlier_.clear();
    num_inliers_ = 0;
    
    // TODO: which one shall we use? residuals or loss?
    std::deque<Dtype> const& used_cache = residual_cache_;
    
    // keep the losses of the most recent loss_cache_size_ many examples in the cache
    int k = used_cache.size();
    //printf("loss cache currently stores [%d] examples\n", k);
    if (k > loss_cache_size_ - outer_num_) {
      // make room for more recent examples
      int num_removed = k - (loss_cache_size_ - outer_num_);
      //printf("num_removed = [%d]\n", num_removed);
      for (int i = 0; i < num_removed; ++i) {
	loss_cache_.pop_front();
	residual_cache_.pop_front();
      }
    }
    
    // compute losses and residual values
    for (int i = 0; i < outer_num_; ++i) {
      tmp_loss = 0.0;
      tmp_residual = 0.0;
      for (int j = 0; j < inner_num_; j++) {
	const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	DCHECK_GE(label_value, 0);
	DCHECK_LT(label_value, prob_.shape(softmax_axis_));
	Dtype p = prob_data[i * dim + label_value * inner_num_ + j];
	tmp_loss -= log(std::max(p, Dtype(FLT_MIN)));    
	tmp_residual += (1.0 - p);
      }
      loss_cache_.push_back(tmp_loss); 
      residual_cache_.push_back(tmp_residual);
    }    
            
    // compute the percentile value from all examples currently in the cache
    int num_cached = used_cache.size();
    for (int i = 0; i < num_cached; ++i) {
      sorted_cache_.push_back(used_cache[i]);
      //printf("cached value: [%.4f]\n", used_cache[i]);
    }
    Dtype percentile_val;
    // TODO: magic number!
    const float percentile = 0.9;
    // TODO: magic number!
    const float percentile_chi_squared_factor = 0.607456739; // 0.95: 0.510310363, 0.9: 0.607456739, 0.5: 1.4826
    if (num_cached % 2 == 0) {
      int percentile_idx;
      percentile_val = 0.0;
      // first element
      percentile_idx = num_cached * percentile - 1;
      CHECK_GT(percentile_idx, 0);
      CHECK_LT(percentile_idx, num_cached - 1);
      std::nth_element(sorted_cache_.begin(), sorted_cache_.begin() + percentile_idx, sorted_cache_.end());
      percentile_val += sorted_cache_[percentile_idx];
      // second element
      percentile_idx += 1;
      std::nth_element(sorted_cache_.begin(), sorted_cache_.begin() + percentile_idx, sorted_cache_.end());
      percentile_val += sorted_cache_[percentile_idx];
      // average
      percentile_val /= 2.0;
      //printf("percentile_val (even size): [%.4f]\n", percentile_val);
    }
    else {
      int percentile_idx = num_cached * percentile;
      CHECK_GT(percentile_idx, 0);
      CHECK_LT(percentile_idx, num_cached);
      std::nth_element(sorted_cache_.begin(), sorted_cache_.begin() + percentile_idx, sorted_cache_.end());
      percentile_val = sorted_cache_[percentile_idx];
      printf("percentile_val (odd size): [%.4f]\n", percentile_val);
    }
    
    // check for the examples added in the current batch whether or not they are outliers
    int min_i = num_cached - outer_num_;
    Dtype stdev = percentile_chi_squared_factor * percentile_val; 
    Dtype outlier_threshold = stdev_z_value_ * stdev;
    //printf("outlier_threshold: [%.4f]\n", outlier_threshold);
    for (int i = min_i; i < num_cached; ++i) {
      bool is_outlier = used_cache[i] > outlier_threshold;
      is_outlier_.push_back(is_outlier);
      if (!is_outlier) {
	// an example may contribute to the loss only if it is no outlier
	loss += loss_cache_[i];
	++num_inliers_;
      }
    }
    if (num_inliers_ < outer_num_) {
      printf("num_inliers_: [%d] (of [%d])\n", num_inliers_, outer_num_);
    }
    
    // if the inlier count is 0, the loss is zero and the count is set to 1 only to avoid division by 0
    count = (num_inliers_ > 0) ? num_inliers_ * inner_num_ : 1;
  }
  if (method_ == 3) {
    is_outlier_.clear();    
    num_inliers_ = 0;
    
    ++batch_count_;
    //printf("batch_count: [%d]", batch_count_);
    
    Dtype outlier_threshold = 1.0 - sqrt(1.0 - retained_inlier_fraction_);
    //printf("outlier_threshold: [%.4f]\n", outlier_threshold);
    
    // compute losses and residual values
    for (int i = 0; i < outer_num_; ++i) {
      tmp_loss = 0.0;
      tmp_residual = 0.0;
      for (int j = 0; j < inner_num_; j++) {
	const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	DCHECK_GE(label_value, 0);
	DCHECK_LT(label_value, prob_.shape(softmax_axis_));
	Dtype p = prob_data[i * dim + label_value * inner_num_ + j];
	tmp_loss -= log(std::max(p, Dtype(FLT_MIN)));    
	tmp_residual += (1.0 - p);
      }

      bool is_outlier = (batch_count_ >= min_batch_count_) && (tmp_residual > outlier_threshold);
      is_outlier_.push_back(is_outlier);
      if (!is_outlier) {
	// an example may contribute to the loss only if it is no outlier
	loss += tmp_loss;
	++num_inliers_;
      }
      else {
	printf("residual: [%.4f] (exceeds outlier_threshold: [%.4f])\n", tmp_residual, outlier_threshold);	
      }
    }    
	  
    if (num_inliers_ < outer_num_) {
      printf("num_inliers_: [%d] (of [%d])\n", num_inliers_, outer_num_);
    }
    
    // if the inlier count is 0, the loss is zero and the count is set to 1 only to avoid division by 0
    count = (num_inliers_ > 0) ? num_inliers_ * inner_num_ : 1;
  }
  
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
    
    if (method_ == 1) {
      for (int k = 0; k < outer_num_; ++k) {
	int i = outliers_[k].second;
	if (k < num_ignored_) {
	  for (int j = 0; j < inner_num_; ++j) {    
	    for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
	      bottom_diff[i * dim + c * inner_num_ + j] = 0;
	    }
	  }
	  //printf("ignored gradient of example [%d]\n", i);
	}
	else {
	  for (int j = 0; j < inner_num_; ++j) {    
	    const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	    bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
	    ++count;
	  }
	  //printf("added gradient of example [%d]\n", i);
	}
      }
    }
    if (method_ == 2 || method_ == 3) {
      for (int i = 0; i < outer_num_; ++i) {
	if (is_outlier_[i]) {
	  for (int j = 0; j < inner_num_; ++j) {    
	    for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
	      bottom_diff[i * dim + c * inner_num_ + j] = 0;
	    }
	  }
	  //printf("ignored gradient of example [%d]\n", i);
	}
	else {
	  for (int j = 0; j < inner_num_; ++j) {    
	    const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	    bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
	  }
	  //printf("added gradient of example [%d]\n", i);
	}
      }      
      // if the inlier count is 0, the loss is zero and the count is set to 1 only to avoid division by 0
      count = (num_inliers_ > 0) ? num_inliers_ * inner_num_ : 1;     
    }
    
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
