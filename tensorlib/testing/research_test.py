import tensorflow as tf
import numpy as np
import time
import cv2
import tensorlib as lib


class ResearchTest(tf.test.TestCase):

    def test_network_weights(self):
        inputs = lib.Input((224, 224, 3), batch_size=2)
        with lib.arg_scope(lib.research.resnet_beta_arg_scope()):
            net = lib.research.ResNet_V1_50_beta(num_classes=1001, global_pool=True)
        # net = lib.research.MobileNetV2(3, 10)
        net.train()
        _ = net(inputs)
        weights = set()
        for w in net.weights:
            if w in weights:
                print('repeat', w)
            else:
                weights.add(w)
                print(w)

    def test_residual_block(self):
        start = time.time()
        inputs = lib.Input((54, 54, 64), batch_size=2)
        net = lib.research.Bottleneck(128 * 4, 128, stride=2,
                                      downsample=lib.layers.Conv2D(
                                          128 * 4, kernel_size=1,
                                          strides=2, use_bias=False))
        outputs = net(inputs)
        print(time.time() - start)
        print(outputs)
        self.assertListEqual(outputs.get_shape().as_list(), [2, 27, 27, 128 * 4])
        writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
        writer.close()

    def test_resnet(self):
        start = time.time()
        inputs = lib.Input((224, 224, 3))
        with lib.arg_scope(lib.research.resnet_beta_arg_scope()):
            model = lib.research.ResNet_V1_101_beta(
                (224, 224, 3), is_training=False,
                global_pool=True, num_classes=1001)
        outputs = model(inputs)
        print(time.time() - start)
        node = getattr(outputs, '_anchor')[0]
        self.assertEqual(node.layer.name, 'resnet_v1_101/')
        self.assertListEqual(outputs.get_shape().as_list(), [1, 1001])
        writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
        writer.close()

    def test_resnet_101_classification(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = './image/butterfly.jpeg'
        with self.test_session() as sess:
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img = img[..., ::-1]
            img = img[np.newaxis, ...]
            img = tf.convert_to_tensor(img)
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            with lib.arg_scope(lib.research.resnet_beta_arg_scope()):
                model = lib.research.ResNet_V1_101_beta(
                    (224, 224, 3), is_training=False,
                    global_pool=True, num_classes=1001)
            output = model(image)
            model.load_weights('./checkpoint/resnet_v1_beta/resnet_v1_101/model.ckpt')
            self.assertEqual(output.get_shape(), (1, 1001))
            sess.run(tf.global_variables_initializer())
            predict = sess.run(output)[0]
            print(np.argmax(predict))
            writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph", tf.get_default_graph())
            writer.close()
            self.assertEqual(np.argmax(predict), 323)

    def test_resnet_50_classification(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = './image/butterfly.jpeg'
        with self.test_session() as sess:
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img = img[..., ::-1]
            img = img[np.newaxis, ...]
            img = tf.convert_to_tensor(img)
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            with lib.arg_scope(lib.research.resnet_beta_arg_scope()):
                model = lib.research.ResNet_V1_50_beta(
                    (224, 224, 3), is_training=False,
                    global_pool=True, num_classes=1001)
            outputs = model(image)
            model.load_weights('../research/resnet_v1_beta/checkpoint/resnet_v1_50/model.ckpt')
            self.assertListEqual(outputs.get_shape().as_list(), [1, 1001])
            sess.run(tf.global_variables_initializer())
            predict = sess.run(outputs)[0]
            print(np.argmax(predict))
            # writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph", tf.get_default_graph())
            # writer.close()
            self.assertEqual(np.argmax(predict), 323)

    def test_mobilenet_v2_forward(self):
        start = time.time()
        inputs = lib.Input((224, 224, 3))
        with lib.arg_scope(lib.research.resnet_beta_arg_scope()):
            model = lib.research.MobileNetV2(
                3, num_classes=1001,
                multiplier=0.25)
        outputs = model(inputs)
        print(time.time() - start)
        self.assertListEqual(outputs.get_shape().as_list(), [1, 1001])
        writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
        writer.close()

    def test_mobilenet_v2_classification(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        path = './image/butterfly.jpeg'
        with self.test_session() as sess:
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img = img[..., ::-1]
            img = img[np.newaxis, ...]
            img = tf.convert_to_tensor(img)
            image = tf.image.convert_image_dtype(img, dtype=tf.float32)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            start = time.time()
            model = lib.research.MobileNetV2(in_channels=3, trainable=False)
            # for layer in model.layers(skip_self=True):
            #     print(layer.name, isinstance(layer, Network))
            outputs = model(image)
            print(time.time() - start)
            self.assertListEqual(outputs.get_shape().as_list(), [1, 1001])
            node = getattr(outputs, '_anchor')[0]
            print(node)
            model.load_weights('../research/mobilenet/checkpoints/mobilenet_v2_1.0.ckpt')
            sess.run(tf.global_variables_initializer())
            predict = sess.run(outputs)[0]
            self.assertEqual(np.argmax(predict), 323)
            # writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
            # writer.close()

    def test_fpn(self):
        fpn = lib.research.FPNWrapper(
            network=lib.research.ResNet_V1_18_beta(base_only=True),
            extract_blocks=['pool1', 'block1/unit_2/lite_bottleneck_v1/add',
                            'block2/unit_2/lite_bottleneck_v1/add',
                            'block3/unit_2/lite_bottleneck_v1/add',
                            'block4/unit_2/lite_bottleneck_v1/add'])
        print(fpn(tf.ones((1, 224, 224, 3))))
        for child in fpn.children():
            print(child, child.name)
        # writer = tf.summary.FileWriter("D:/GeekGank/workspace/graph/model_graph", tf.get_default_graph())
        # writer.close()
