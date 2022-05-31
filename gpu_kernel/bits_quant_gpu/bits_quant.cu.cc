#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

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


// Define the CUDA kernel.
template <typename T>
__global__ void BitsQuantCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = Quantization(__ldg(in + i));
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void BitsQuantFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // Launch the cuda kernel.
  //
  // See core/util/gpu_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  BitsQuantCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct BitsQuantFunctor<GPUDevice, float>;
template struct BitsQuantFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA