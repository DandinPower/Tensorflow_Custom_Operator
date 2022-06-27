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
int shift_normal,detect_normal,remove_normal,inject_normal;
int shift_approximate,detect_approximate,remove_approximate,inject_approximate;

void init(){
  shift_normal = 0;
  shift_approximate = 0;
  detect_normal = 0;
  detect_approximate = 0;
  remove_normal = 0;
  remove_approximate = 0;
  inject_normal = 0;
  inject_approximate = 0;
}

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

void PermutationWrite(float x, float y){
    std::string binary_x = float_to_bin(x);
    std::string binary_y = float_to_bin(y);
    int Q_x = CountQ(binary_x);
    int Q_y = CountQ(binary_y);
    shift_normal += 2 * (FLOAT_BITS_LENGTH + 1) + Max((Q_x - Q_y), 0);
    detect_normal += FLOAT_BITS_LENGTH;
    remove_normal += Max((Q_x - Q_y), 0);
    inject_normal += Max((Q_y - Q_x), 0);
    shift_approximate += 2 * (FLOAT_BITS_LENGTH + 1);
    detect_approximate += FLOAT_BITS_LENGTH;
}

//當前面的tensor比後面大的時候(代表需要刪除)
void RemoveOld(float x){
    std::string binary_x = float_to_bin(x);
    int Q_x = CountQ(binary_x);
    shift_normal += 2 * (FLOAT_BITS_LENGTH);
    remove_normal += Q_x;
}

//當後面的tensor比前面大的時候(代表需要重新寫入)
void WriteNew(float y){
    std::string binary_y = float_to_bin(y);
    int Q_y = CountQ(binary_y);
    shift_normal += 2 * (FLOAT_BITS_LENGTH);
    inject_normal += Q_y;
    shift_approximate += 2 * (FLOAT_BITS_LENGTH + 1);
    detect_approximate += FLOAT_BITS_LENGTH;

}

REGISTER_OP("CountSkrm")
    .Input("to_zero: float")
    .Input("to_zero2: float")
    .Input("shape: int64")
    .Output("zeroed: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });

class CountSkrmOp : public OpKernel {
 public:
  explicit CountSkrmOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    init();
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_shape = context->input(2);
    auto input = input_tensor.flat<float>();
    auto input2 = input_tensor2.flat<float>();
    int N1 = input.size();
    int N2 = input2.size();
    int MaxN = Max(N1, N2);
    for (int i = 0; i< MaxN; i++){
      if ((N1-1) < i) WriteNew(input2(i));
      else if((N2 -1) < i) RemoveOld(input(i));
      else PermutationWrite(input(i),input2(i));
    }
    //std::cout << shift_normal << ' ' << detect_normal << ' ' << remove_normal << ' ' << inject_normal << std::endl;
    //std::cout << shift_approximate << ' ' << detect_approximate << ' ' << remove_approximate << ' ' << inject_approximate << std::endl;

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64>();
    output_flat(0) = shift_normal;
    output_flat(1) = detect_normal;
    output_flat(2) = remove_normal;
    output_flat(3) = inject_normal;
    output_flat(4) = shift_approximate;
    output_flat(5) = detect_approximate;
    output_flat(6) = remove_approximate;
    output_flat(7) = inject_approximate;
  }
};

REGISTER_KERNEL_BUILDER(Name("CountSkrm").Device(DEVICE_CPU), CountSkrmOp);