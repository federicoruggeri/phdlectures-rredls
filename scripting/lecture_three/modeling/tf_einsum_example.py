import numpy as np
import tensorflow as tf


@tf.function
def test_transpose(x):
    x_T = tf.transpose(x, [0, 2, 1])
    ein_T = tf.einsum('ijk->ikj', x)
    return x_T, ein_T


@tf.function
def test_sum(x):
    sum_x = tf.reduce_sum(x)
    ein_sum = tf.einsum('ijk->', x)
    return sum_x, ein_sum


@tf.function
def test_axis_sum(x):
    sum_x = tf.reduce_sum(x, axis=-1)
    ein_sum = tf.einsum('ijk->ij', x)
    return sum_x, ein_sum


@tf.function
def test_mm(x):
    mm_x = tf.matmul(x, x, transpose_a=True)
    ein_mm = tf.einsum('ijk,ikl->ijl', tf.einsum('ijk->ikj', x), x)
    return mm_x, ein_mm


if __name__ == '__main__':
    assignment_matrix = np.array([0., 0.,
                                  0., 1.,
                                  0., 0.,
                                  0., 0.,
                                  1., 0.]).reshape(5, 2).astype(np.float32)[np.newaxis, :, :]

    # Transpose
    print(test_transpose(x=assignment_matrix))
    print(test_sum(x=assignment_matrix))
    print(test_axis_sum(x=assignment_matrix))
    print(test_mm(x=assignment_matrix))