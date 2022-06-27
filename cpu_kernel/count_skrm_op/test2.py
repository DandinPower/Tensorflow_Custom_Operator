import tensorflow as tf
kernal_module = tf.load_op_library('./count_skrm.so')
tensorA = tf.constant([[[1.2,2.3],[3.5,-4.1]],[[1.2,2.3],[3.5,-4.1]]])
print(kernal_module.count_skrm(tensorA))
