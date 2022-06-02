load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "bits_quant.so",
    srcs = ["bits_quant.h", "bits_quant.cc"],
    gpu_srcs = ["bits_quant.cu.cc", "bits_quant.h"],
)