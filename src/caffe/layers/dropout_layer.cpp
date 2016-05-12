// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  impl_ = this->layer_param_.dropout_param().impl();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);  
  //printf("DEBUGGING: bottom[0] [%s]\n", bottom[0]->shape_string().c_str());
  //printf("DEBUGGING: top[0] [%s]\n", top[0]->shape_string().c_str());
  //exit(1);
  if (impl_ == 1) {
    LOG(INFO) << "WARNING: using custom impl for Dropout";
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  if (impl_ == 0) {
    rand_vec_.Reshape(bottom[0]->shape());
  }
  else {
    // randomize only over the batch and channels, i.e., N x C x 1 x 1
    int N = bottom[0]->shape(0);
    int C = bottom[0]->shape(1);
    rand_vec_.Reshape(N, C, 1, 1);
    //LOG(INFO) << "Dropout operates on " << N << " batch examples";
    //LOG(INFO) << "Dropout operates on " << C << " channels";
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();  
  if (impl_ == 0) {  
    const int count = bottom[0]->count();
    if (this->phase_ == TRAIN) {
      // Create random numbers
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      for (int i = 0; i < count; ++i) {
	top_data[i] = bottom_data[i] * mask[i] * scale_;
      }
    } else {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
  }
  else {    
    // we treat all batches the same; also, in a fully-convolutional network, it should not matter
    // where the filter is applied, so if a neuron activiation is disabled, it is along all spatial 
    // dimensions
    // @TODO: wouldn't it be better to determine scale_ by the number of non-zero mask elements, 
    // instead of using the threshold? the original implementation might scale by too much/too small
    if (this->phase_ == TRAIN) {
      const int N = bottom[0]->shape(0);
      const int C = bottom[0]->shape(1);
      const int H = bottom[0]->shape(2);
      const int W = bottom[0]->shape(3);
      // Create random numbers
      caffe_rng_bernoulli(N * C, 1. - threshold_, mask);
      // @TODO: not very elegant, but should be what we want
      int i = 0;
      for (int n = 0; n < N; ++n) {
	for (int c = 0; c < C; ++c) {
	  for (int h = 0; h < H; ++h) {
	    for (int w = 0; w < W; ++w) {
	      top_data[i] = bottom_data[i] * mask[n * C + c] * scale_;
	      ++i;
	    }
	  }
	}
      }
    } else {
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }      
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (impl_ == 0) {
      if (this->phase_ == TRAIN) {
	const unsigned int* mask = rand_vec_.cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i) {
	  bottom_diff[i] = top_diff[i] * mask[i] * scale_;
	}
      } else {
	caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }
    }
    else {
      if (this->phase_ == TRAIN) {
	const unsigned int* mask = rand_vec_.cpu_data();
	const int N = bottom[0]->shape(0);
	const int C = bottom[0]->shape(1);
	const int H = bottom[0]->shape(2);
	const int W = bottom[0]->shape(3);
	// @TODO: not very elegant, but should be what we want
	int i = 0;
	for (int n = 0; n < N; ++n) {
	  for (int c = 0; c < C; ++c) {
	    for (int h = 0; h < H; ++h) {
	      for (int w = 0; w < W; ++w) {
		bottom_diff[i] = top_diff[i] * mask[n * C + c] * scale_;
		++i;
	      }
	    }
	  }
	}	
      } else {
	caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }      
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
