import tensorflow as tf
import numpy as np
import os
import random
import threading
from datetime import datetime
import sys
from collections import namedtuple


class MetaData(namedtuple('MetaData',
                          ['image', 'ann'])):
    pass


class ImageDecoder(object):

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()
        self._encoded_image = tf.placeholder(dtype=tf.string)
        self._decode_image = tf.image.decode_image(self._encoded_image, channels=3)

    def decode_image(self, encoded_img):
        image = self._sess.run(self._decode_image,
                               feed_dict={self._encoded_image: encoded_img})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class Encoder(object):

    def encode(self, metas):
        """
        :param metas: list of instances of `MetaData`
        :return: a dictionary {'feature': feature} or {'feature_list': feature},
         feature is a dict contains all annotation in TensorFlow's Feature form
        """
        raise NotImplementedError


def int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[value.encode('utf-8') if type(value) == str else value]))


def float_feature(value):
    """Wrapper for inserting a float Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def float_feature_list(values):
    """Wrapper for inserting a float FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def to_sequence_example(meta, encoder):
    """
    Builds a SequenceExample proto for an image-detection pair.
    :param meta: An Metadata object.
    :param encoder: An instance of Encoder
    :return: a SequenceExample proto for an image-detection pair.
    """
    features = encoder.encode(meta)
    context = tf.train.Features(feature=features['feature'])
    if 'feature_list' in features:
        feature_lists = tf.train.FeatureLists(feature_list=features['feature_list'])
        example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)
    else:
        example = tf.train.Example(
            features=context)
    return example


def assign_tasks(root_dir, metas, encoder, num_shards, num_threads=4, division='train'):
    root_dir = os.path.join(root_dir, division)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    assert division in ['train', 'val', 'test', 'demo']
    random.seed(12345)
    random.shuffle(metas)
    count = len(metas)
    num_shards = min(num_shards, count)
    num_threads = min(num_shards, num_threads)
    spacing = np.linspace(0, count, num_threads + 1).astype(np.int)
    spacing = np.expand_dims(spacing, axis=1)
    batches = np.concatenate((spacing[:-1], spacing[1:]), axis=1)
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    # Launch a thread for each batch.
    threads = []
    print("Launch %d threads for spacings: %s" % (num_threads, batches))
    for thread_index in range(num_threads):
        args = (thread_index, batches, division,
                metas, encoder, num_shards, root_dir)
        t = threading.Thread(target=process_fn, args=args)
        t.start()
        threads.append(t)
        # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d meta dataset in dataset set '%s'." %
          (datetime.now(), len(metas), division))


def process_fn(thread_index, ranges, division, metas,
               encoder, num_shards, root_dir):
    """
    metas divide into batches, each batch divides into shards
    :param thread_index: batch index, thread index
    :param ranges: each batch meta dataset indices
    :param division: name of this meta dataset, ['train', 'val', 'test']
    :param metas: meta dataset list
    :param encoder: encoder for meta
    :param num_shards: number of shard
    :param root_dir: top dir name
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)
    indices = ranges[thread_index]
    shard_ranges = np.linspace(indices[0], indices[1],
                               num_shards_per_batch + 1).astype(int)
    num_metas_in_thread = indices[1] - indices[0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (division, shard, num_shards)

        output_file = os.path.join(root_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        metas_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in metas_in_shard:
            meta = metas[i]
            sequence_example = to_sequence_example(meta, encoder)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("\n%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_metas_in_thread))
                sys.stdout.flush()

        writer.close()
        print("\n%s [thread %d]: Wrote %d meta dataset to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print("\n%s [thread %d]: Wrote %d meta dataset to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()
