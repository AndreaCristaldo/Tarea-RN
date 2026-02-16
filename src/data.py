import tensorflow as tf

def load_fashion_mnist_flat():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0
    return x_train, y_train, x_test, y_test
