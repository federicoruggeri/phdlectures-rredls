import os
from functools import partial
from typing import Iterator, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.utility.benchmarking_utility import fix_seed, simulate_iterator
from scripting.utility.logging_utility import Logger


class Preprocessor:

    def __init__(
            self
    ):
        self.tokenizer = Tokenizer()

    def setup(
            self,
            train_df: pd.DataFrame
    ):
        texts = train_df['Sentence'].values
        texts = list(map(lambda t: self.preprocess_text(t), texts))
        self.tokenizer.fit_on_texts(texts=texts)

    def preprocess_text(
            self,
            text: str
    ) -> str:
        text = text.lower()
        text = text.strip()
        return text

    def parse_inputs(
            self,
            index: tf.Tensor,
            df: pd.DataFrame
    ) -> [tf.Tensor, tf.Tensor]:
        texts = df.iloc[index.numpy()]['Sentence']
        labels = df.iloc[index.numpy()]['Label']

        texts = list(map(lambda t: self.preprocess_text(t), [texts]))
        texts = self.tokenizer.texts_to_sequences(texts)[0]
        return texts, labels

    def get_steps(
            self,
            data: np.ndarray
    ) -> int:
        num_batches = int(np.ceil(len(data) / batch_size))
        return num_batches

    def light_iterator(
            self,
            df: pd.DataFrame,
    ) -> Iterator:
        for idx in range(df.shape[0]):
            yield idx

    # Note: the tf.data.Dataset.from_generator and self._make_iterator must be executed by the same python process!
    def make_iterator(
            self,
            df: pd.DataFrame,
            batch_size: int = 32,
            shuffle: bool = False,
            prefetch: bool = False,
    ):
        data_generator = partial(self.light_iterator, df=df)
        data = tf.data.Dataset.from_generator(generator=data_generator,
                                              output_types=tf.int32)
        if shuffle:
            data = data.shuffle(buffer_size=100)

        data = data.map(map_func=lambda idx: tf.py_function(func=partial(self.parse_inputs, df=df),
                                                            inp=[idx],
                                                            Tout=[tf.int32, tf.int32]),
                        num_parallel_calls=tf.data.AUTOTUNE)

        data = data.padded_batch(batch_size=batch_size,
                                 padded_shapes=([None], []))

        if prefetch:
            data = data.prefetch(buffer_size=tf.data.AUTOTUNE)

        return data


if __name__ == '__main__':
    # Settings
    samples_amount = -1
    batch_size = 32
    random_seed = 42
    simulation_time = 0.001
    prefetch = True
    info_basename = 'tf_data_pipeline_gen_pyfunc_{0}_{1}.npy'
    save_info = True

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
    timing_info['train'], memory_info['train'] = simulate_iterator(iterator=train_iterator,
                                                                   steps=train_steps,
                                                                   description='train iterator',
                                                                   simulation_time=simulation_time)

    val_steps = preprocessor.get_steps(data=val_df['Sentence'].values)
    val_iterator = partial(preprocessor.make_iterator,
                           df=val_df,
                           prefetch=prefetch,
                           batch_size=batch_size)
    timing_info['val'], memory_info['val'] = simulate_iterator(iterator=val_iterator,
                                                               steps=val_steps,
                                                               description='val iterator',
                                                               simulation_time=simulation_time)

    test_steps = preprocessor.get_steps(data=test_df['Sentence'].values)
    test_iterator = partial(preprocessor.make_iterator,
                            df=test_df,
                            prefetch=prefetch,
                            batch_size=batch_size)
    timing_info['test'], memory_info['test'] = simulate_iterator(iterator=test_iterator,
                                                                 steps=test_steps,
                                                                 description='test iterator',
                                                                 simulation_time=simulation_time)

    logger.info(f'''Times:
        {timing_info}
        ''')
    logger.info(f'''Memory usage:
        {memory_info}
        ''')
    if save_info:
        np.save(os.path.join(info_save_dir, info_basename.format(prefetch, 'timing')), timing_info)
        np.save(os.path.join(info_save_dir, info_basename.format(prefetch, 'memory')), memory_info)
