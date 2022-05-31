import tensorflow as tf
with tf.device("/gpu:0"):
	kernel =  tf.load_op_library('./bits_quant.so')
	print(kernel.bits_quant([[1, 2], [3, 4]]).numpy())