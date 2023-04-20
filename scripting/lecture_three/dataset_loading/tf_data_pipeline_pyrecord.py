import os
from functools import partial
from typing import List, AnyStr

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.utility.benchmarking_utility import fix_seed, simulate_iterator
from scripting.utility.logging_utility import Logger
from scripting.utility.tensorflow_utility import create_int_feature


class InputFeature:

    def __init__(
            self,
            token_ids: np.ndarray,
            label_id: int
    ):
        self.token_ids = token_ids
        self.label_id = label_id

    @classmethod
    def get_mappings(
            cls,
    ):
        mappings = {
            'token_ids': tf.io.VarLenFeature(tf.int64),
            'label_id': tf.io.FixedLenFeature([1], tf.int64),
        }
        return mappings

    @classmethod
    def get_feature_records(
            cls,
            feature
    ):
        features = dict()
        features['token_ids'] = create_int_feature(feature.token_ids)
        features['label_id'] = create_int_feature([feature.label_id])
        return features

    @classmethod
    def get_dataset_selector(cls):
        def _selector(record):
            x = record['token_ids']
            y = record['label_id']
            return x, y

        return _selector


class Preprocessor:

    def __init__(
            self,
            serialization_dir: AnyStr
    ):
        self.tokenizer = Tokenizer()
        self.serialization_path = os.path.join(serialization_dir, 'tf_pyrecord_data')

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
            texts: np.ndarray,
            labels: np.ndarray
    ) -> [List[str], np.ndarray]:
        texts = list(map(lambda t: self.preprocess_text(t), texts))
        texts = self.tokenizer.texts_to_sequences(texts)

        return texts, labels

    def get_steps(
            self,
            data: np.ndarray
    ) -> int:
        num_batches = int(np.ceil(len(data) / batch_size))
        return num_batches

    def serialize_data(
            self,
            df: pd.DataFrame,
            suffix: str
    ):
        texts, labels = df['Sentence'].values, df['Label'].values
        texts, labels = self.parse_inputs(texts=texts, labels=labels)

        Logger.get_logger(__name__).info('Serializing data...')
        with tf.io.TFRecordWriter(self.serialization_path + f'_{suffix}') as writer:
            with tqdm(total=len(texts)) as pbar:
                for idx, (text, label) in enumerate(zip(texts, labels)):
                    feature = InputFeature(token_ids=text,
                                           label_id=label)
                    feature_records = InputFeature.get_feature_records(feature=feature)
                    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_records))
                    writer.write(tf_example.SerializeToString())

                    pbar.set_description(desc=f'Serializing example {idx}/{len(texts)}')
                    pbar.update(1)

    def decode_record(
            self,
            record,
            name_to_features
    ):
        """
        TPU does not support int64
        """
        example = tf.io.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        example['token_ids'] = tf.sparse.to_dense(example['token_ids'])
        example['label_id'] = tf.reshape(example['label_id'], ())

        return example

    # Note: the tf.data.Dataset.from_generator and self._make_iterator must be executed by the same python process!
    def make_iterator(
            self,
            df: pd.DataFrame,
            suffix: str,
            batch_size: int = 32,
            shuffle: bool = False,
            prefetch: bool = False,
    ):
        # Serialize only if needed!
        if not os.path.isfile(self.serialization_path + f'_{suffix}'):
            self.serialize_data(df=df,
                                suffix=suffix)

        data = tf.data.TFRecordDataset(self.serialization_path + f'_{suffix}')

        if shuffle:
            data = data.shuffle(buffer_size=100)

        data = data.map(lambda record: self.decode_record(record,
                                                          name_to_features=InputFeature.get_mappings()),
                        num_parallel_calls=tf.data.AUTOTUNE)
        data = data.padded_batch(batch_size=batch_size,
                                 padded_shapes=({'token_ids': [None],
                                                 'label_id': []}))

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
    info_basename = 'tf_data_pipeline_pyrecord_{0}_{1}.npy'
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
                             df=train_df,
                             suffix='train',
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
                           suffix='val',
                           prefetch=prefetch,
                           batch_size=batch_size)
    timing_info['val'], memory_info['val'] = simulate_iterator(iterator=val_iterator,
                                                               steps=val_steps,
                                                               description='val iterator',
                                                               simulation_time=simulation_time)

    test_steps = preprocessor.get_steps(data=test_df['Sentence'].values)
    test_iterator = partial(preprocessor.make_iterator,
                            df=test_df,
                            suffix='test',
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
