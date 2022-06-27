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
std::string float_to_bin(float x){
    char bitsString[FLOAT_BITS_LENGTH];
    int fl = *(int*)&x;
    for (int i = 0; i < sizeof(float) * 8; ++i)
    bitsString[FLOAT_BITS_LENGTH -1 -i] = ((1 << i) & fl) != 0 ? '1' : '0';
    return bitsString;
}

int CountQ(std::string x){
    int total = 0;
    for (int i=0; i<FLOAT_BITS_LENGTH; i++){
        if (x[i] == '1') total++;
    }
    return total;
}

int Max(int x, int y){
    if (x >= y) return x;
    else return y;
}

std::vector<int> PermutationWrite(float x, float y){
    std::string binary_x = float_to_bin(x);
    std::string binary_y = float_to_bin(y);
    std::cout << binary_x << std::endl;
    std::cout << binary_y << std::endl;
    int Q_x = CountQ(binary_x);
    int Q_y = CountQ(binary_y);
    int shift = 2 * (FLOAT_BITS_LENGTH + 1) + Max((Q_x - Q_y), 0);
    int detect = FLOAT_BITS_LENGTH;
    int remove = Max((Q_x - Q_y), 0);
    int inject = Max((Q_y - Q_x), 0);
    std::vector<int> answer;
    answer.push_back(shift);
    answer.push_back(detect);
    answer.push_back(remove);
    answer.push_back(inject);
    return answer;
}

REGISTER_OP("CountSkrm")
    .Input("to_zero: float")
    .Input("to_zero2: float")
    .Output("zeroed: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class CountSkrmOp : public OpKernel {
 public:
  explicit CountSkrmOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    auto input = input_tensor.flat<float>();
    auto input2 = input_tensor2.flat<float>();
    std::vector answer = PermutationWrite(input(0),input2(1));
    std::cout << answer[0] << ' ' << answer[1] << ' ' << answer[2] << ' ' << answer[3] << std::endl;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<float>();
    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0.0;
    }
    // Preserve the first input value if possible.
    if (N > 0) output_flat(0) = input(0);
  }
};

REGISTER_KERNEL_BUILDER(Name("CountSkrm").Device(DEVICE_CPU), CountSkrmOp);