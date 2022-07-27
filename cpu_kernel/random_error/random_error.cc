#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <random>
#include <climits>
#include <string.h>
#include <string>
#include <time.h>
#include <cmath>
#include <vector>

using namespace tensorflow;
#define FLOAT_BITS_LENGTH 32

std::random_device rd;
std::default_random_engine eng(rd());


//根據要翻轉的bits數取得rate
long GetFlipRate(int num){
    long result = 0;
    for (int i=0; i<num; i++) result += pow(2, i);
    return result;
}

//根據給定的rate來翻轉bits
int FlipBits(float* x, long rate){
    int flip_nums = 0;
    int* address = (int *)x; 
    *address ^= rate;
    for (int i = 0; i < FLOAT_BITS_LENGTH; ++i) flip_nums += ((1 << i) & rate) != 0 ? 1 : 0;
    return flip_nums;
}

//根據範圍取得rate
long GetRateByRange(long start, long end){
    std::uniform_int_distribution<long> distr(start, end);
    return distr(eng);
}

REGISTER_OP("RandomError")
    .Input("origin_tensor: float")
    .Output("error_tensor: float")
    .Attr("error_rate: float = 0.25")
    .Attr("start: int = 0")
    .Attr("end: int = 32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

class RandomErrorOp : public OpKernel {
 public:
  explicit RandomErrorOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("error_rate", &error_rate));
    OP_REQUIRES_OK(context, context->GetAttr("start", &start));
    OP_REQUIRES_OK(context, context->GetAttr("end", &end));
  }

  void Compute(OpKernelContext* context) override {
    std::cout << "range " << start << " " << end << std::endl;
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    int N = input.size();
    int total_error_bits_nums = N * FLOAT_BITS_LENGTH * error_rate;
    std::cout << "total_error_bits_nums " << total_error_bits_nums << std::endl;
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();
    long bits_rate_start = GetFlipRate(start);
    long bits_rate_end = GetFlipRate(end);
    std::cout << "bits_rate_range " << bits_rate_start << " " << bits_rate_end << std::endl;
    for (int i=0; i<N; i++){
      float origin = input(i);
      if (total_error_bits_nums > 0){
        long random_flip_rate = GetRateByRange(bits_rate_start, bits_rate_end);
        std::cout << "random " << random_flip_rate << std::endl;
        total_error_bits_nums -= FlipBits(&origin, random_flip_rate);
      }
      std::cout << "total_error_bits_nums after flip " << total_error_bits_nums << std::endl;
      output_flat(i) = origin;
    }
  }
  private:
    int start;
    int end;
    float error_rate;
};

REGISTER_KERNEL_BUILDER(Name("RandomError").Device(DEVICE_CPU), RandomErrorOp);