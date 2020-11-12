import tensorflow as tf
import tensorlib as lib
tf.random.set_random_seed(666)
import keras


class FunctionTest(tf.test.TestCase):

    def test_dot(self):
        x = tf.random.normal((4, 3, 5))
        y = tf.random.normal((4, 2, 3))
        a = lib.engine.batch_dot(x, y, [1, 2])
        b = keras.backend.batch_dot(x, y, axes=[1, 2])
        with self.cached_session() as sess:
            c, d = sess.run([a, b])
            self.assertAllEqual(c, d)
