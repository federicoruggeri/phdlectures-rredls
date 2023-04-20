import os
from typing import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.utility.benchmarking_utility import fix_seed, simulate_iterator
from scripting.utility.logging_utility import Logger
from functools import partial


class Preprocessor:

    def __init__(
            self
    ):
        self.tokenizer = get_tokenizer(tokenizer='basic_english')
        self.vocab = None

    def setup(
            self,
            train_df: pd.DataFrame
    ):
        texts = train_df['Sentence'].values
        texts = list(map(lambda t: self.preprocess_text(t), texts))
        counter = Counter()
        for text in texts:
            counter.update(self.tokenizer(text))
        self.vocab = Vocab(counter)

    def preprocess_text(
            self,
            text: str
    ) -> str:
        text = text.lower()
        text = text.strip()
        return text

    def parse_inputs(
            self,
            df: pd.DataFrame,
    ) -> [np.ndarray, np.ndarray]:
        texts = df['Sentence'].values
        labels = df['Label'].values

        texts = list(map(lambda t: self.preprocess_text(t), texts))
        texts = list(map(lambda t: [self.vocab[token] for token in self.tokenizer(t)], texts))
        return np.array(texts, dtype=object), labels

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
            shuffle: bool = False
    ) -> Iterator:
        texts, labels = self.parse_inputs(df=df)

        assert len(texts) == len(labels), f'Inconsistent number of texts and labels'

        num_batches = self.get_steps(data=texts)
        for batch_idx in range(num_batches):
            if shuffle:
                batch_indexes = np.random.randint(low=0, high=len(texts), size=batch_size)
            else:
                start_index = batch_idx * batch_size
                end_index = min(batch_idx * batch_size + batch_size, len(texts))
                batch_indexes = np.arange(start_index, end_index)

            assert len(batch_indexes) <= batch_size

            batch_texts = texts[batch_indexes].tolist()
            text_max_length = max(list(map(lambda t: len(t), batch_texts)))

            batch_texts = map(lambda t: t + [0] * (text_max_length - len(t)), batch_texts)

            yield batch_texts, labels[batch_indexes]


if __name__ == '__main__':
    # Settings
    samples_amount = -1
    batch_size = 32
    random_seed = 42
    simulation_time = 0.001
    info_basename = 'torch_naive_pipeline_{}.npy'
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
                             shuffle=True)
    timing_info['train'], memory_info['train'] = simulate_iterator(iterator=train_iterator,
                                                                   steps=train_steps,
                                                                   description='train iterator',
                                                                   simulation_time=simulation_time)

    val_steps = preprocessor.get_steps(data=val_df['Sentence'].values)
    val_iterator = partial(preprocessor.make_iterator,
                           df=val_df,
                           batch_size=batch_size)
    timing_info['val'], memory_info['val'] = simulate_iterator(iterator=val_iterator,
                                                               steps=val_steps,
                                                               description='val iterator',
                                                               simulation_time=simulation_time)

    test_steps = preprocessor.get_steps(data=test_df['Sentence'].values)
    test_iterator = partial(preprocessor.make_iterator,
                            df=test_df,
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
        np.save(os.path.join(info_save_dir, info_basename.format('timing')), timing_info)
        np.save(os.path.join(info_save_dir, info_basename.format('memory')), memory_info)
