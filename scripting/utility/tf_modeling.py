import abc

import tensorflow as tf
from typing import Callable, Iterator
from tqdm import tqdm
import time
from scripting.utility.logging_utility import Logger
from memory_profiler import memory_usage
import numpy as np
from scripting.utility.printing_utility import prettify_statistics
import os


class M_LSTM(tf.keras.Model):

    def __init__(
            self,
            embedding_dimension,
            vocab_size,
            lstm_weights,
            answer_units,
            l2_regularization=0.,
            **kwargs
    ):
        super(M_LSTM, self).__init__(**kwargs)
        self.input_embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=embedding_dimension,
                                                         mask_zero=True,
                                                         name='input_embedding')
        # LSTM blocks
        self.lstm_block = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_weights))
        self.final_block = tf.keras.layers.Dense(units=answer_units,
                                                 kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))

    def call(
            self,
            input_ids,
            training=False
    ):
        # [bs, N, d']
        input_emb = self.input_embedding(input_ids,
                                         training=training)

        # [bs, d']
        encoded_inputs = self.lstm_block(input_emb,
                                         training=training)

        # [bs, d'']
        answer = self.final_block(encoded_inputs,
                                  training=training)
        return answer


class TFModelWrapper:

    @abc.abstractmethod
    def loss_op(
            self,
            x,
            targets,
            training=False
    ):
        pass

    @abc.abstractmethod
    def train_op(
            self,
            x,
            y
    ):
        pass

    @abc.abstractmethod
    def batch_fit(
            self,
            x,
            y
    ):
        pass


class TFTrainer:

    def __init__(
            self,
            epochs: int = 3
    ):
        self.epochs = epochs

    def _run(
            self,
            model: TFModelWrapper,
            train_data_iterator: Callable[[], Iterator],
            steps: int
    ):
        for epoch in range(self.epochs):
            epoch_train_iterator = train_data_iterator()
            train_loss = {}
            with tqdm(total=steps) as pbar:
                for step in range(steps):
                    batch = next(epoch_train_iterator)
                    batch_loss, batch_loss_info = model.batch_fit(*batch)
                    batch_loss_info = {f'train_{key}': item.numpy() for key, item in batch_loss_info.items()}
                    batch_loss_info['train_loss'] = batch_loss.numpy()

                    # Update epoch loss
                    for key, item in batch_loss_info.items():
                        if key in train_loss:
                            train_loss[key] += item
                        else:
                            train_loss[key] = item

                    pbar.set_description(desc=f'Training -- Epoch {epoch + 1}')
                    pbar.update(1)

            train_loss = {key: item / steps for key, item in train_loss.items()}
            train_loss = {key: float('{:.2f}'.format(value)) for key, value in train_loss.items()}
            train_loss['epoch'] = epoch + 1
            Logger.get_logger(__name__).info(f'{os.linesep}{prettify_statistics(train_loss)}')

    def run(
            self,
            model: TFModelWrapper,
            train_data_iterator: Callable[[], Iterator],
            steps: int
    ):
        Logger.get_logger(__name__).info('Running training phase...')
        start_time = time.perf_counter()
        mem_usage = memory_usage((self._run, (), {"model": model,
                                                  "train_data_iterator": train_data_iterator,
                                                  "steps": steps}))
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return total_time, np.mean(mem_usage)
