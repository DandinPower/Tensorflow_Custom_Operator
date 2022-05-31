#ifndef BITS_QUANT_H_
#define BITS_QUANT_H_
#include <iostream>
#include <bitset>
#include <climits>
#include <string.h>
#include <string>
#include<ctime>
#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct BitsQuantFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct BitsQuantFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif BITS_QUANT_H_