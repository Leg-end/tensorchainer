import tensorflow as tf
import numpy as np
import cv2
from tensorlib.research.mobilenet.mobile_v2 import MobileNetV2


class MobileNetTest(tf.test.TestCase):

    def testInfer(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = './test_img/butterfly.jpeg'
        with self.test_session() as sess:
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img = img[..., ::-1]
            img = img[np.newaxis, ...]
            img = tf.convert_to_tensor(img)
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            model = MobileNetV2(in_channels=3, trainable=False)
            output = model(image)
            for w in model.weights:
                print(w)
            model.load_weights('./checkpoints/mobilenet_v2_1_0.h5')
            self.assertEqual(output.get_shape(), (1, 1001))
            predict = sess.run(output)[0]
            self.assertEqual(np.argmax(predict), 323)


if __name__ == '__main__':
    tf.test.main()
