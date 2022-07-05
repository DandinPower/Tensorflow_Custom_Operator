import tensorflow as tf
kernal_module = tf.load_op_library('./count_skrm.so')
tensorShape = tf.zeros([8],tf.int64)

#test case 1 (前面資料量較大)
tensorA = tf.constant([1.2, 3.1])
tensorB = tf.constant([-0.3])
print(kernal_module.count_skrm(tensorA,tensorB,tensorShape))

#test case 2 (資料量一樣大)
tensorA = tf.constant([1.2, 3.1])
tensorB = tf.constant([-0.3, 4.1])
print(kernal_module.count_skrm(tensorA,tensorB,tensorShape))

#test case 3 (後面資料量較大)
tensorA = tf.constant([1.1])
tensorB = tf.constant([-0.3, 3.1])
print(kernal_module.count_skrm(tensorA,tensorB,tensorShape))


