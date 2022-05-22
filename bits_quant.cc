#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("BitsQuant")
    .Input("to_zero: float32")
    .Output("zeroed: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

union
{
    float input; 
    int   output;
} data;
int counter = 15;

float Quantization(float x){
    counter = 15;
    data.input = x;
    std::bitset<sizeof(float) * CHAR_BIT> bits(data.output);
    std::string bits_string = bits.to_string();
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
    int sign = (bits_string[0]=='1')?-1:1;    
    std::bitset<8> exp_8(bits_string,1,8);
    int exp = exp_8.to_ulong()-127;
    std::string fraction_23 = bits_string.substr(9,23);
    float ff=1,bb=1;
    for(int i=0;i<23;i++){
        bb/=2;
        if(fraction_23[i]=='1') ff+=bb;
    }
    if (exp >= 0) ff=ff*(1<<exp)*sign;
    else ff=ff/(1<<(-exp))*sign;
    return ff;
}

class BitsQuantOp : public OpKernel {
 public:
  explicit BitsQuantOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<floats32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = Quantization(input(i));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("BitsQuant").Device(DEVICE_CPU), BitsQuantOp);