#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype> 
__global__ void DropoutForward_FCN(const int count, 
    const int N, const int C, const int H, const int W, 
    const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {  
    const int n = index / (C * H * W); // which batch element; 0 <= n < N
    const int c = (index - n * (C * H * W)) / (H * W); // which channel; 0 <= c < C
    out[index] = in[index] * (mask[n * C + c] > threshold) * scale;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  if (impl_ == 0) {
    const int count = bottom[0]->count();
    if (this->phase_ == TRAIN) {
      unsigned int* mask =
	  static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
      caffe_gpu_rng_uniform(count, mask);
      // set thresholds
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	  count, bottom_data, mask, uint_thres_, scale_, top_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(count, bottom_data, top_data);
    }
  }
  else {
    const int count = bottom[0]->count();
    if (this->phase_ == TRAIN) {
      const int N = bottom[0]->shape(0);
      const int C = bottom[0]->shape(1);
      const int H = bottom[0]->shape(2);
      const int W = bottom[0]->shape(3);
      unsigned int* mask =
	  static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
      caffe_gpu_rng_uniform(N * C, mask);
      // set thresholds
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutForward_FCN<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	  count, N, C, H, W, bottom_data, mask, uint_thres_, scale_, top_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(count, bottom_data, top_data);
    }     
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
__global__ void DropoutBackward_FCN(const int count, 
    const int N, const int C, const int H, const int W, 
    const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, count) {  
    const int n = index / (C * H * W); // which batch element; 0 <= n < N
    const int c = (index - n * (C * H * W)) / (H * W); // which channel; 0 <= c < C
    out_diff[index] = in_diff[index] * scale * (mask[n * C + c] > threshold);
  } 
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (impl_ == 0) {
      if (this->phase_ == TRAIN) {
	const unsigned int* mask =
	    static_cast<const unsigned int*>(rand_vec_.gpu_data());
	const int count = bottom[0]->count();
	// NOLINT_NEXT_LINE(whitespace/operators)
	DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
	  CAFFE_CUDA_NUM_THREADS>>>(
	    count, top_diff, mask, uint_thres_, scale_, bottom_diff);
	CUDA_POST_KERNEL_CHECK;
      } else {
	caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }
    }
    else {
      if (this->phase_ == TRAIN) {
	const int count = bottom[0]->count();
	const int N = bottom[0]->shape(0);
	const int C = bottom[0]->shape(1);
	const int H = bottom[0]->shape(2);
	const int W = bottom[0]->shape(3);
	const unsigned int* mask =
	    static_cast<const unsigned int*>(rand_vec_.gpu_data());
	// NOLINT_NEXT_LINE(whitespace/operators) 
	DropoutBackward_FCN<Dtype><<<CAFFE_GET_BLOCKS(count),
	  CAFFE_CUDA_NUM_THREADS>>>(
	    count, N, C, H, W, top_diff, mask, uint_thres_, scale_, bottom_diff);
	CUDA_POST_KERNEL_CHECK;
      } else {
	caffe_copy(top[0]->count(), top_diff, bottom_diff);
      } 
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
