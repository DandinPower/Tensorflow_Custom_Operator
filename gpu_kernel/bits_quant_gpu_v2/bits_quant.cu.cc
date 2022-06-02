#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <iostream>
#include <bitset>
#include <climits>
#include <string.h>
#include <string>
#include<ctime>
using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

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