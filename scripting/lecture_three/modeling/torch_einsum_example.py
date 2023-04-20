import numpy as np
import torch as th


def test_transpose(x):
    x = th.tensor(x)
    x_T = th.permute(x, [0, 2, 1])
    ein_T = th.einsum('ijk->ikj', x)
    return x_T, ein_T


def test_sum(x):
    x = th.tensor(x)
    sum_x = th.sum(x)
    ein_sum = th.einsum('ijk->', x)
    return sum_x, ein_sum


def test_axis_sum(x):
    x = th.tensor(x)
    sum_x = th.sum(x, dim=-1)
    ein_sum = th.einsum('ijk->ij', x)
    return sum_x, ein_sum


def test_mm(x):
    x = th.tensor(x)
    mm_x = th.matmul(th.permute(x, [0, 2, 1]), x)
    ein_mm = th.einsum('ijk,ikl->ijl', th.einsum('ijk->ikj', x), x)
    return mm_x, ein_mm


if __name__ == '__main__':
    # Settings
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