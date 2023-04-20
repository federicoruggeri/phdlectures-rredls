import numpy as np
import tensorflow as tf


def supervision_loss(
        prob_dist,
        positive_indexes,
        negative_indexes,
        mask_indexes,
        supervision_margin=0.5
):
    padding_amount = positive_indexes.shape[-1]

    # Repeat mask for each positive element in each sample memory
    # Mask_idxs shape: [batch_size, padding_amount]
    # Mask res shape: [batch_size * padding_amount, padding_amount]
    mask_res = tf.tile(mask_indexes, multiples=[1, padding_amount])
    mask_res = tf.reshape(mask_res, [-1, padding_amount, padding_amount])
    mask_res = tf.transpose(mask_res, [0, 2, 1])
    mask_res = tf.reshape(mask_res, [-1, padding_amount])

    # Split each similarity score for a target into a separate sample
    # similarities shape: [batch_size, memory_max_length]
    # positive_idxs shape: [batch_size, padding_amount]
    # gather_nd shape: [batch_size, padding_amount]
    # pos_scores shape: [batch_size * padding_amount, 1]
    pos_scores = tf.gather(prob_dist, positive_indexes, batch_dims=1)
    pos_scores = tf.reshape(pos_scores, [-1, 1])

    # Repeat similarity scores for non-target memories for each positive score
    # similarities shape: [batch_size, memory_max_length]
    # negative_idxs shape: [batch_size, padding_amount]
    # neg_scores shape: [batch_size * padding_amount, padding_amount]
    neg_scores = tf.gather(prob_dist, negative_indexes, batch_dims=1)
    neg_scores = tf.tile(neg_scores, multiples=[1, padding_amount])
    neg_scores = tf.reshape(neg_scores, [-1, padding_amount])

    # Compare each single positive score with all corresponding negative scores
    # [batch_size * padding_amount, padding_amount]
    # [batch_size, padding_amount]
    # [batch_size, 1]
    # Samples without supervision are ignored by applying a zero mask (mask_res)
    hop_supervision_loss = tf.maximum(0., supervision_margin - pos_scores + neg_scores)
    hop_supervision_loss = hop_supervision_loss * tf.cast(mask_res, dtype=hop_supervision_loss.dtype)
    hop_supervision_loss = tf.reshape(hop_supervision_loss, [-1, padding_amount, padding_amount])

    hop_supervision_loss = tf.reduce_sum(hop_supervision_loss, axis=[1, 2])
    normalization_factor = tf.cast(tf.reshape(mask_res, [-1, padding_amount, padding_amount]),
                                   hop_supervision_loss.dtype)
    normalization_factor = tf.reduce_sum(normalization_factor, axis=[1, 2])
    normalization_factor = tf.maximum(normalization_factor, tf.ones_like(normalization_factor))
    hop_supervision_loss = tf.reduce_sum(hop_supervision_loss / normalization_factor)

    # Normalize by number of positive examples
    valid_examples = tf.reduce_sum(mask_indexes, axis=1)
    valid_examples = tf.cast(valid_examples, tf.float32)
    valid_examples = tf.minimum(valid_examples, 1.0)
    valid_examples = tf.reduce_sum(valid_examples)
    valid_examples = tf.maximum(valid_examples, 1.0)
    hop_supervision_loss = hop_supervision_loss / tf.cast(valid_examples, hop_supervision_loss.dtype)
    return hop_supervision_loss


if __name__ == '__main__':
    prob_dist = np.array([
        0.1, 0.1, 0.2, 0.4, 0.1, 0.1,
        0.5, 0.0, 0.0, 0.1, 0.1, 0.3
    ]).reshape(2, 6)
    positive_indexes = np.array([2, 3, 3, 3, 3,
                                 0, 0, 0, 0, 0]).reshape(2, 5)
    negative_indexes = np.array([0, 1, 4, 5, 5,
                                 1, 2, 3, 4, 5]).reshape(2, 5)
    mask_indexes = np.array([1, 1, 0, 0, 0,
                             1, 0, 0, 0, 0]).reshape(2, 5)
    print(supervision_loss(prob_dist=prob_dist,
                           positive_indexes=positive_indexes,
                           negative_indexes=negative_indexes,
                           mask_indexes=mask_indexes))
