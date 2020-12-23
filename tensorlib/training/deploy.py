import os
import tensorflow as tf


__all__ = ['get_frozen_graph', 'load_graph']


def get_frozen_graph(saver,
                     graph,
                     model_dir,
                     output_name='outputs',
                     gpu_devices=None):
    if not os.path.exists(model_dir):
        raise ValueError("Model dir can not be none.")
    del tf.get_collection_ref(tf.GraphKeys.TRAIN_OP)[:]
    if gpu_devices:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    ckpt_path = checkpoint.model_checkpoint_path.replace('\\', '/')
    abs_dir = '/'.join(ckpt_path.split('/')[:-1])
    out_dir = os.path.join(abs_dir, 'frozen_model.pb')
    with tf.Session(graph=graph) as sess:
        sess.run('init')
        saver.restore(sess, ckpt_path)
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=output_name.split(','))
        with tf.gfile.GFile(out_dir, 'wb') as f:
            f.write(graph_def.SerializeToString())
        print("Frozen {:d} ops in the final graph".format(len(graph_def.node)))
    return graph_def


def load_graph(path):
    with tf.gfile.GFile(path, "rb")as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default()as graph:
        tf.import_graph_def(graph_def, input_map=None, name='')  # name="prefix"
    return graph
