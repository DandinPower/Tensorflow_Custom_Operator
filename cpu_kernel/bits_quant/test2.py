import tensorflow as tf
bits_quant_module = tf.load_op_library('./bits_quant.so')
tensorA = tf.constant([[[1.2,2.3],[3.5,-4.1]],[[1.2,2.3],[3.5,-4.1]]])
print(bits_quant_module.bits_quant(tensorA))
