import tensorflow as tf
import numpy as np
from scripting.utility.logging_utility import Logger
import os
from scripting.utility.tensorflow_utility import stable_norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class ExampleModel:

    def __init__(
            self
    ):
        self.coefficient = 1.0

    @tf.function
    def compute_overlap_constraint(
            self,
            assignment_matrix
    ):
        K = assignment_matrix.shape[-1]

        # [bs, K, K]
        root_intensities = tf.matmul(assignment_matrix, assignment_matrix, transpose_a=True)
        root_intensities = root_intensities / stable_norm(root_intensities, axis=[1, 2])[:, None, None]

        # [K, K]
        eye_matrix = tf.eye(K)
        eye_matrix = eye_matrix / stable_norm(eye_matrix)

        # [bs, K, K]
        penalty = root_intensities - eye_matrix[None, :]
        penalty = stable_norm(penalty, axis=[1, 2])
        return tf.reduce_mean(penalty) * self.coefficient

    @tf.function
    def compute_overlap_constraint_with_coefficient(
            self,
            assignment_matrix,
            coefficient
    ):
        K = assignment_matrix.shape[-1]

        # [bs, K, K]
        root_intensities = tf.matmul(assignment_matrix, assignment_matrix, transpose_a=True)
        root_intensities = root_intensities / stable_norm(root_intensities, axis=[1, 2])[:, None, None]

        # [K, K]
        eye_matrix = tf.eye(K)
        eye_matrix = eye_matrix / stable_norm(eye_matrix)

        # [bs, K, K]
        penalty = root_intensities - eye_matrix[None, :]
        penalty = stable_norm(penalty, axis=[1, 2])
        return tf.reduce_mean(penalty) * tf.cast(coefficient, penalty.dtype)


def plot_coefficients(
        values
):
    fig, ax = plt.subplots(1, 1)
    ax.plot(values)
    plt.show()


def compute_constraint(
        assignment_matrix
):
    pairwise_diff = model.compute_overlap_constraint(assignment_matrix=assignment_matrix[np.newaxis, :, :])
    return pairwise_diff.numpy()


def compute_constraint_with_coefficient(
        assignment_matrix,
        coefficient
):
    pairwise_diff = model.compute_overlap_constraint_with_coefficient(
        assignment_matrix=assignment_matrix[np.newaxis, :, :],
        coefficient=coefficient)
    return pairwise_diff.numpy()


def simulate_coefficient_annealing(
        model,
        assignment_matrix: np.ndarray,
        iterations: int = 100
):
    coefficient_values = np.linspace(0, 1.0, iterations)[::-1]
    logger.info('Simulating coefficient annealing...')
    values = []
    with tqdm(total=iterations) as pbar:
        for it in range(iterations):
            constraint = compute_constraint(assignment_matrix=assignment_matrix)
            values.append(constraint)

            # Update coefficient
            model.coefficient = coefficient_values[it]

            pbar.update(1)
            pbar.set_description(desc=f'Simulating coefficient = {model.coefficient}')

    logger.info(f'Constraint values: {values}')
    plot_coefficients(values=values)


def simulate_coefficient_annealing_with_coefficient(
        model,
        assignment_matrix: np.ndarray,
        iterations: int = 100
):
    coefficient_values = np.linspace(0, 1.0, iterations)[::-1]
    logger.info('Simulating coefficient annealing...')
    values = []
    with tqdm(total=iterations) as pbar:
        for it in range(iterations):
            constraint = compute_constraint_with_coefficient(assignment_matrix=assignment_matrix,
                                                             coefficient=model.coefficient)
            values.append(constraint)

            # Update coefficient
            model.coefficient = coefficient_values[it]

            pbar.update(1)
            pbar.set_description(desc=f'Simulating coefficient = {model.coefficient}')

    logger.info(f'Constraint values: {values}')
    plot_coefficients(values=values)


if __name__ == '__main__':
    # Settings
    iterations = 100

    this_path = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(this_path,
                                             os.pardir,
                                             os.pardir,
                                             os.pardir))

    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    Logger.set_log_path(name='logger',
                        log_path=log_dir)
    logger = Logger.get_logger(__name__)

    assignment_matrix = np.array([0., 0.,
                                  0., 1.,
                                  0., 0.,
                                  0., 0.,
                                  1., 0.]).reshape(5, 2).astype(np.float32)

    model = ExampleModel()
    compute_constraint(assignment_matrix=assignment_matrix)
    simulate_coefficient_annealing(assignment_matrix=assignment_matrix,
                                   iterations=iterations,
                                   model=model)

    # Reset model coefficient
    model.coefficient = 1.0
    simulate_coefficient_annealing_with_coefficient(assignment_matrix=assignment_matrix,
                                                    iterations=iterations,
                                                    model=model)
