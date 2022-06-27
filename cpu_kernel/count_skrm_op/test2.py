import tensorflow as tf
kernal_module = tf.load_op_library('./count_skrm.so')
tensorA = tf.constant([1.123])
tensorB = tf.constant([-0.323231])
print(kernal_module.count_skrm(tensorA,tensorB))
