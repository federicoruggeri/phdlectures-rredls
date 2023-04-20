import tensorflow as tf
import os


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def tf_fix_seed(
        seed: int
):
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def stable_norm(x, axis=None):
    """
    tf.norm is not numerically stable when computing gradient (wtf).
    Link: https://datascience.stackexchange.com/questions/80898/tensorflow-gradient-returns-nan-or-inf
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + 1.0e-08)