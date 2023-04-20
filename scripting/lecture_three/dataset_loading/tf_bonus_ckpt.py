import os
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.utility.benchmarking_utility import fix_seed, run_iterator
from scripting.utility.logging_utility import Logger


class Preprocessor:

    def __init__(
            self
    ):
        self.tokenizer = TextVectorization()

    def setup(
            self,
            train_df: pd.DataFrame
    ):
        texts = train_df['Sentence'].values
        data = tf.data.Dataset.from_tensors(texts)
        self.tokenizer.adapt(data=data)

    def parse_inputs(
            self,
            inputs: Dict,
    ) -> [np.ndarray, np.ndarray]:

        text = inputs['Sentence']
        text = self.tokenizer(tf.expand_dims(text, 0))[0]   # expand to add 'batch' dimension

        return text, inputs['Label']

    def get_steps(
            self,
            data: np.ndarray
    ) -> int:
        num_batches = int(np.ceil(len(data) / batch_size))
        return num_batches

    def make_iterator(
            self,
            df: pd.DataFrame,
            batch_size: int = 32,
            shuffle: bool = False,
            prefetch: bool = False,
    ):
        data = tf.data.Dataset.from_tensor_slices(dict(df))

        if shuffle:
            data = data.shuffle(buffer_size=100)

        data = data.map(map_func=self.parse_inputs,
                        num_parallel_calls=tf.data.AUTOTUNE)

        data = data.padded_batch(batch_size=batch_size,
                                 padded_shapes=([None], []))

        if prefetch:
            data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return iter(data)


if __name__ == '__main__':
    # Settings
    samples_amount = -1
    batch_size = 32
    random_seed = 42
    simulation_time = 0.001
    prefetch = True
    info_basename = 'tf_data_pipeline_slices_{0}_{1}.npy'
    save_info = True
    batches_to_take = 3

    this_path = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(this_path,
                                             os.pardir,
                                             os.pardir,
                                             os.pardir))

    info_save_dir = os.path.join(base_dir,
                                 'scripting',
                                 'lecture_three',
                                 'runtime_data')

    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    Logger.set_log_path(name='logger',
                        log_path=log_dir)
    logger = Logger.get_logger(__name__)

    # Seeding
    fix_seed(seed=random_seed)

    # Loading
    train_df, val_df, test_df = load_ibm2015_dataset(samples_amount=samples_amount)

    # Pre-processing pipeline
    preprocessor = Preprocessor()
    preprocessor.setup(train_df=train_df)

    timing_info = {}
    memory_info = {}

    train_steps = preprocessor.get_steps(data=train_df['Sentence'].values)
    train_iterator = partial(preprocessor.make_iterator,
                             df=train_df,
                             batch_size=batch_size,
                             prefetch=prefetch,
                             shuffle=True)
    train_iterator = train_iterator()

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=train_iterator)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(info_save_dir, 'tf_data_pipeline_slices_ckpt'), max_to_keep=2)

    run_iterator(iterator=train_iterator,
                 takes=batches_to_take)

    save_path = manager.save()

    run_iterator(iterator=train_iterator,
                 takes=batches_to_take)

    ckpt.restore(manager.latest_checkpoint)

    run_iterator(iterator=train_iterator,
                 takes=batches_to_take)
