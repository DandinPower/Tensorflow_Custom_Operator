#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <climits>
#include <string.h>
#include <string>
#include <time.h>
#include <cmath>
#include <vector>

using namespace tensorflow;
#define FLOAT_BITS_LENGTH 32

int error_bits_nums = 0;

//將浮點數->001010010101....
std::string float_to_bin(float x){
    char bitsString[FLOAT_BITS_LENGTH];
    int fl = *(int*)&x;
    for (int i = 0; i < sizeof(float) * 8; ++i)
    bitsString[FLOAT_BITS_LENGTH -1 -i] = ((1 << i) & fl) != 0 ? '1' : '0';
    return bitsString;
}

//將001010010101.. -> float
float StringToFloat(std::string bitsString){
    int sign = (bitsString[0]=='1')?-1:1;    
    int exp = -127;
    float bb = 1;
    for(int i=8;i>0;i--){
        if(bitsString[i]=='1') exp+=bb;
        bb *= 2;
    }
    bb = 1;
    float ff=1;
    for(int i=9;i<32;i++){
        bb/=2;
        if(bitsString[i]=='1') ff+=bb;
    }
    ff = ff * pow(2,exp) * sign;
    return ff;
}

//計算010010101... 裡有幾個1
int CountQ(std::string x){
    int total = 0;
    for (int i=0; i<FLOAT_BITS_LENGTH; i++){
        if (x[i] == '1') total++;
    }
    return total;
}

//取最大值
int Max(int x, int y){
    if (x >= y) return x;
    else return y;
}

REGISTER_OP("RandomError")
    .Input("origin_tensor: float")
    .Output("error_tensor: float")
    .Attr("error_type: int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

class RandomErrorOp : public OpKernel {
 public:
  explicit RandomErrorOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    init();
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    int N = input.size();
    int total_bits_nums = N * FLOAT_BITS_LENGTH;
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64>();
    for (int i=0; i<N; i++){
      std::string bits = float_to_bin(input(i));
      
      
      
      output_flat(i) = NULL;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("RandomError").Device(DEVICE_CPU), RandomErrorOp);