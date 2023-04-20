import numpy as np
import tensorflow as tf


@tf.function
def ptk_overlap_constraint(pooling_matrix):
    # pooling_matrix is [bs, N, K]
    K = pooling_matrix.shape[-1]

    # [bs, K, N]
    pooling_matrix = tf.transpose(pooling_matrix, [0, 2, 1])

    # [bs, K, 1, N] - [bs, 1, K, N]
    # [bs, K, K]
    penalty = tf.reduce_sum(tf.abs(pooling_matrix[:, :, None, :] - pooling_matrix[:, None, :, :]), axis=-1)
    penalty = tf.nn.relu(1.0 - penalty)

    diag_mask = tf.ones_like(penalty) - tf.eye(penalty.shape[-1], dtype=tf.float32)[None, :, :]
    penalty *= diag_mask

    # [bs,]
    denominator = K ** 2 - K
    denominator = tf.maximum(tf.cast(denominator, tf.float32), 1.0)

    penalty = tf.reduce_sum(penalty, axis=(-1, -2)) / denominator
    return penalty


@tf.function
def show_broadcasting(pooling_matrix):
    pooling_matrix = tf.transpose(pooling_matrix, [0, 2, 1])

    # [bs, K, K]
    broadcasted_penalty = tf.reduce_sum(tf.abs(pooling_matrix[:, :, None, :] - pooling_matrix[:, None, :, :]), axis=-1)
    return broadcasted_penalty


if __name__ == '__main__':
    pooling_matrix = np.array([1, 1, 0, 0, 0,
                               0, 0, 1, 1, 1]).reshape(2, 5).transpose().astype(np.float32)

    penalty = ptk_overlap_constraint(pooling_matrix=pooling_matrix[np.newaxis, :, :])
    print(f'Penalty: {penalty}')
    print(f'Broadcasted penalty: {show_broadcasting(pooling_matrix[np.newaxis, :, :])}')
