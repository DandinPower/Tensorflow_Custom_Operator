import tensorflow as tf
import numpy as np

class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.kernel = tf.load_op_library('./bits_quant.so')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=64,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=64,activation=tf.nn.relu)
        self.output1 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.kernel.bits_quant(x)
        x = self.dense2(x)
        output = self.output1(x)
        return output

inputs = tf.constant([1,2,3,4,5,6,7,8,9,10])
y = tf.constant([0,1,0,0,0,0,0,0,0,0])
model = TestModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

with tf.GradientTape() as tape:
    logits = model(inputs, training=True)
    loss_value = loss_fn(y, logits)
grads = tape.gradient(loss_value, model.trainable_weights)
print(grads[0])
optimizer.apply_gradients(zip(grads, model.trainable_weights))