#include "bits_quant.h"
#include <iostream>
#include <climits>
#include <string.h>
#include <string>
#include<ctime>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("BitsQuant")
    .Attr("T: numbertype")
    .Input("input: T")
    .Output("input_times_two: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

std::string FloatToString(float x) {
    char bitsString[32];
    int fl = *(int*)&x;
    for (int i = 0; i < sizeof(float) * 8; ++i)
    bitsString[31-i] = ((1 << i) & fl) != 0 ? '1' : '0';
    return bitsString;
}

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
int counter = 15;

float Quantization(float x){
    counter = 15;
    std::string bits_string = FloatToString(x);
    for (int i=0; i<32; i++){
        if (bits_string[i] == '1') {
            if (counter > 0) {
                bits_string[i] = '1';
                counter--;
            }
            else bits_string[i] = '0';
        }
        else{
            bits_string[i] = '0';
        }
    }
    if (counter > 0){
        for (int i=31;i>=0;i--){
            if (bits_string[i] == '0'){
                bits_string[i] = '1';
                counter--;
                if (counter == 0) break;
            }
        }
    }
    return StringToFloat(bits_string);
}

// CPU specialization of actual computation.
template <typename T>
struct BitsQuantFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = Quantization(in[i]);
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class BitsQuantOp : public OpKernel {
 public:
  explicit BitsQuantOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    BitsQuantFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BitsQuant").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      BitsQuantOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class BitsQuantFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("BitsQuant").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      BitsQuantOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA