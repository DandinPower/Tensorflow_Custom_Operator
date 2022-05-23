import tensorflow as tf
with tf.device("/gpu:0"):
	kernel =  tf.load_op_library('./kernel_example.so')
	print(kernel.example([[1, 2], [3, 4]]).numpy())