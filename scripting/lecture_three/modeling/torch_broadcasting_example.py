import numpy as np
import torch as th


def ptk_overlap_constraint(
        pooling_matrix
):
    pooling_matrix = th.tensor(pooling_matrix)

    # pooling_matrix is [bs, N, K]
    K = pooling_matrix.shape[-1]

    # [bs, K, N]
    pooling_matrix = th.permute(pooling_matrix, [0, 2, 1])

    # [bs, K, 1, N] - [bs, 1, K, N]
    # [bs, K, K]
    penalty = th.sum(th.abs(pooling_matrix[:, :, None, :] - pooling_matrix[:, None, :, :]), dim=-1)
    penalty = th.relu(1.0 - penalty)

    diag_mask = th.ones_like(penalty) - th.eye(penalty.shape[-1], dtype=th.float32)[None, :, :]
    penalty *= diag_mask

    # [bs,]
    denominator = K ** 2 - K
    denominator = th.tensor(np.maximum(denominator, 1.0))

    penalty = th.sum(penalty, dim=(-1, -2)) / denominator
    return penalty


def show_broadcasting(
        pooling_matrix
):
    pooling_matrix = th.tensor(pooling_matrix)

    pooling_matrix = th.permute(pooling_matrix, [0, 2, 1])

    # [bs, K, K]
    broadcasted_penalty = th.sum(th.abs(pooling_matrix[:, :, None, :] - pooling_matrix[:, None, :, :]), dim=-1)
    return broadcasted_penalty


if __name__ == '__main__':
    pooling_matrix = np.array([1, 1, 0, 0, 0,
                               0, 0, 1, 1, 1]).reshape(2, 5).transpose().astype(np.float32)

    penalty = ptk_overlap_constraint(pooling_matrix=pooling_matrix[np.newaxis, :, :])
    print(f'Penalty: {penalty}')
    print(f'Broadcasted penalty: {show_broadcasting(pooling_matrix[np.newaxis, :, :])}')
