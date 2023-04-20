import torch as th
import tensorflow as tf
import os


def test_gather_dim0():
    params = th.arange(15).reshape(5, 3)
    indexes = th.tensor([0, 1, 2,
                         0, 0, 4]).reshape(2, -1)
    res = th.gather(params, dim=0, index=indexes)
    print(f'Params: {os.linesep}{params}')
    print(f'Indexes: {os.linesep}{indexes}')
    print(f'Gather: {os.linesep}{res}')


def test_gather_dim1():
    params = th.arange(15).reshape(5, 3)
    indexes = th.tensor([0, 1, 2,
                         0, 0, 2]).reshape(2, -1)
    res = th.gather(params, dim=1, index=indexes)
    print(f'Params: {os.linesep}{params}')
    print(f'Indexes: {os.linesep}{indexes}')
    print(f'Gather: {os.linesep}{res}')


def test_gather_nd_dim0():
    params = tf.reshape(tf.range(15), [5, 3])
    indexes = tf.reshape(tf.constant([[0, 0], [1, 1], [2, 2],
                                      [0, 0], [0, 1], [4, 2]]), [2, -1, 2])
    res = tf.gather_nd(params, indices=indexes)
    print(f'Params: {os.linesep}{params}')
    print(f'Indexes: {os.linesep}{indexes}')
    print(f'Gather: {os.linesep}{res}')


if __name__ == '__main__':
    test_gather_dim0()
    print('*' * 50)
    print('*' * 50)
    test_gather_dim1()
    print('*' * 50)
    test_gather_nd_dim0()
