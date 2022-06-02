TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++14 -shared bits_quant.cu.cc bits_quant.h bits_quant.cc -o bits_quant.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/cuda/include -L /usr/local/cuda/lib64/ -O2



- 將其整個資料夾放置tensorflow/tensorflow/core/user_ops/裡