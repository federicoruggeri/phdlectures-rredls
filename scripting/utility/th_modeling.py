import abc
import os
import time
from typing import Callable, Iterator

import numpy as np
import torch as th
from memory_profiler import memory_usage
from tqdm import tqdm

from scripting.utility.logging_utility import Logger
from scripting.utility.printing_utility import prettify_statistics


class M_LSTM(th.nn.Module):

    def __init__(
            self,
            embedding_dimension,
            vocab_size,
            lstm_weights,
            answer_units,
    ):
        super(M_LSTM, self).__init__()

        self.input_embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                               embedding_dim=embedding_dimension)

        # LSTM blocks
        self.lstm_block = th.nn.LSTM(input_size=embedding_dimension,
                                     hidden_size=lstm_weights,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True)
        self.final_block = th.nn.Linear(in_features=lstm_weights * 2,
                                        out_features=answer_units)
        self.final_activation = th.nn.ReLU()

    def forward(
            self,
            input_ids
    ):
        # [bs, N, d']
        input_emb = self.input_embedding(input_ids)

        # [bs, d']
        _, (h_n, c_n) = self.lstm_block(input_emb)
        encoded_inputs = th.permute(h_n, [1, 0, 2])
        encoded_inputs = encoded_inputs.reshape(encoded_inputs.shape[0], -1)

        # [bs, d'']
        answer = self.final_block(encoded_inputs)
        answer = self.final_activation(answer)
        return answer


class THModelWrapper:

    def __init__(
            self,
    ):
        self.model = None

    @abc.abstractmethod
    def build_model(
            self,
            *args,
            **kwargs
    ):
        pass

    @abc.abstractmethod
    def loss_op(
            self,
            x,
            targets
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


class ThTrainer:

    def __init__(
            self,
            epochs: int = 3
    ):
        self.epochs = epochs

    def _run(
            self,
            model: THModelWrapper,
            train_data_iterator: Callable[[], Iterator],
            steps: int
    ):
        model.model.train()
        for epoch in range(self.epochs):
            epoch_train_iterator = train_data_iterator()
            train_loss = {}
            with tqdm(total=steps) as pbar:
                for step in range(steps):
                    batch = next(epoch_train_iterator)
                    batch_loss, batch_loss_info = model.batch_fit(*batch)
                    batch_loss_info = {f'train_{key}': item.detach().numpy() for key, item in batch_loss_info.items()}
                    batch_loss_info['train_loss'] = batch_loss.detach().numpy()

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
            model: THModelWrapper,
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
