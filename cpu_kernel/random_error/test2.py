import tensorflow as tf
kernal_module = tf.load_op_library('./random_error.so')

a = tf.constant([1.2321, -7.19312])

b = kernal_module.random_error(a, 0.25, 0, 32)
print(b)

b = kernal_module.random_error(a, 0.25, 0, 23)
print(b)

b = kernal_module.random_error(a, 0.01, 25, 32)
print(b)
