from tensorflow import keras
import tensorflow as tf
import tensorlib as lib
tf.random.set_random_seed(666)
data_format = 'channels_last'


class ConsistenceTest(tf.test.TestCase):

    def valid_equal(self, a, b):
        name = getattr(a, '_anchor')[0].layer.name
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            o, k_o = sess.run([a, b])
        self.assertAllEqual(o, k_o, msg=name)

    def test_dense(self):
        inputs = tf.random.normal((2, 2))
        dense = lib.layers.Dense(units=4, kernel_initializer='ones')
        k_dense = keras.layers.Dense(units=4, kernel_initializer='ones')
        out = dense(inputs)
        k_out = k_dense(inputs)
        self.valid_equal(out, k_out)

    def test_k_conv1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        k_layer = keras.layers.Conv1D(
            filters=3, kernel_size=3,
            kernel_initializer='ones',
            padding='same',
            data_format=data_format)
        k_out = k_layer(inputs)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(k_out))

    def test_conv1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.Conv1D(
            out_channels=3, kernel_size=3,
            kernel_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.Conv1D(
            filters=3, kernel_size=3,
            kernel_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_conv2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.Conv2D(
            out_channels=3, kernel_size=3,
            kernel_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.Conv2D(
            filters=3, kernel_size=3,
            kernel_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_conv3d(self):
        inputs = tf.random.normal((1, 5, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.Conv3D(
            out_channels=3, kernel_size=3,
            kernel_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.Conv3D(
            filters=3, kernel_size=3,
            kernel_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_separable_conv1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.SeparableConv1D(
            out_channels=3, kernel_size=3,
            depthwise_initializer='ones',
            pointwise_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.SeparableConv1D(
            filters=3, kernel_size=3,
            depthwise_initializer='ones',
            pointwise_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)
        
    def test_separable_conv2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.SeparableConv2D(
            out_channels=3, kernel_size=3,
            depthwise_initializer='ones',
            pointwise_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.SeparableConv2D(
            filters=3, kernel_size=3,
            depthwise_initializer='ones',
            pointwise_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)
        
    def test_depthwise_conv2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.DepthWiseConv2D(
            kernel_size=3,
            depthwise_initializer='ones',
            data_format=data_format)
        k_layer = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            depthwise_initializer='ones',
            padding='same',
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)
        
    def test_flatten(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.Flatten(data_format=data_format)
        k_layer = keras.layers.Flatten(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_maxpool1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.MaxPool1D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.MaxPool1D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_avgpool1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.AvgPool1D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.AvgPool1D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_maxpool1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalMaxPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalMaxPool1D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_avgpool1d(self):
        inputs = tf.random.normal((1, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalAvgPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalAvgPool1D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_maxpool2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.MaxPool2D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.MaxPool2D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_avgpool2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.AvgPool2D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.AvgPool2D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_maxpool2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalMaxPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalMaxPool2D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_avgpool2d(self):
        inputs = tf.random.normal((1, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalAvgPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalAvgPool2D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_maxpool3d(self):
        inputs = tf.random.normal((1, 5, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.MaxPool3D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.MaxPool3D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_avgpool3d(self):
        inputs = tf.random.normal((1, 5, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.AvgPool3D(
            kernel_size=3, strides=2,
            data_format=data_format)
        k_layer = keras.layers.AvgPool3D(
            pool_size=3, strides=2,
            data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_maxpool3d(self):
        inputs = tf.random.normal((1, 5, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalMaxPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalMaxPool3D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)

    def test_global_avgpool3d(self):
        inputs = tf.random.normal((1, 5, 5, 5, 3))
        if data_format == 'channels_first':
            inputs = lib.engine.transpose_to_channels_first(inputs)
        layer = lib.layers.GlobalAvgPool(keepdims=False, data_format=data_format)
        k_layer = keras.layers.GlobalAvgPool3D(data_format=data_format)
        out = layer(inputs)
        k_out = k_layer(inputs)
        self.valid_equal(out, k_out)
