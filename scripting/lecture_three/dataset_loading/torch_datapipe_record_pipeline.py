import os
from functools import partial
from typing import Iterator, List, Tuple, AnyStr

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import TFRecordLoader, FileLister, FileOpener
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.utility.benchmarking_utility import fix_seed, simulate_iterator
from scripting.utility.logging_utility import Logger


class Preprocessor:

    def __init__(
            self,
            serialization_dir: AnyStr,
    ):
        self.tokenizer = get_tokenizer(tokenizer='basic_english')
        self.vocab = None
        self.serialization_path = os.path.join(serialization_dir, 'tf_pyrecord_fast_data')

    def setup(
            self,
            train_df: pd.DataFrame
    ):
        texts = train_df['Sentence'].values
        texts = map(lambda t: self.tokenizer(self.preprocess_text(t)), texts)
        self.vocab = build_vocab_from_iterator(iterator=texts, specials=['<UNK>'])
        self.vocab.set_default_index(self.vocab['<UNK>'])

    def preprocess_text(
            self,
            text: str
    ) -> str:
        text = text.lower()
        text = text.strip()
        return text

    def parse_inputs(
            self,
            input_data: Tuple[str, int]
    ) -> [List[int], int]:
        text, label = input_data
        text = self.preprocess_text(text=text)
        tokens = self.vocab(self.tokenizer(text))
        return tokens, label

    def get_steps(
            self,
            data: np.ndarray
    ) -> int:
        num_batches = int(np.ceil(len(data) / batch_size))
        return num_batches

    def batch_data(
            self,
            input_batch
    ):
        texts, labels = [], []
        for item in input_batch:
            texts.append(item['token_ids'])
            labels.append(item['label_id'])

        texts = pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.int32)
        return {'token_ids': texts,
                'label_ids': labels}

    def make_iterator(
            self,
            suffix: str,
            batch_size: int = 32,
            num_workers: int = 4,
            shuffle: bool = False
    ) -> Iterator:
        base_dir = os.path.dirname(self.serialization_path)
        basename = os.path.basename(self.serialization_path + f'_{suffix}')
        data = FileLister(root=base_dir,
                          masks=basename + '_*')
        data = FileOpener(data, mode='b')
        data = TFRecordLoader(data)

        if shuffle:
            data = data.shuffle(buffer_size=100)

        data = data.sharding_filter()
        data = DataLoader(data,
                          shuffle=shuffle,  # ensures the previous shuffle works (??)
                          batch_size=batch_size,
                          num_workers=num_workers,
                          collate_fn=self.batch_data)
        return iter(data)


if __name__ == '__main__':
    # Settings
    samples_amount = -1
    batch_size = 32
    num_workers = 4
    random_seed = 42
    simulation_time = 0.001
    info_basename = 'torch_datapipe_record_pipeline_{}.npy'
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
    preprocessor = Preprocessor(serialization_dir=info_save_dir)
    preprocessor.setup(train_df=train_df)

    timing_info = {}
    memory_info = {}

    train_steps = preprocessor.get_steps(data=train_df['Sentence'].values)
    train_iterator = partial(preprocessor.make_iterator,
                             suffix='train',
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)
    timing_info['train'], memory_info['train'] = simulate_iterator(iterator=train_iterator,
                                                                   steps=train_steps,
                                                                   description='train iterator',
                                                                   debug=True,
                                                                   simulation_time=simulation_time)

    val_steps = preprocessor.get_steps(data=val_df['Sentence'].values)
    val_iterator = partial(preprocessor.make_iterator,
                           suffix='val',
                           num_workers=num_workers,
                           batch_size=batch_size)
    timing_info['val'], memory_info['val'] = simulate_iterator(iterator=val_iterator,
                                                               steps=val_steps,
                                                               description='val iterator',
                                                               simulation_time=simulation_time)

    test_steps = preprocessor.get_steps(data=test_df['Sentence'].values)
    test_iterator = partial(preprocessor.make_iterator,
                            suffix='test',
                            num_workers=num_workers,
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
