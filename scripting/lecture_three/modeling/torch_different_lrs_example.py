import os
from functools import partial

import torch as th

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.lecture_three.dataset_loading.torch_datapipe_pipeline import Preprocessor
from scripting.utility.benchmarking_utility import fix_seed
from scripting.utility.logging_utility import Logger
from scripting.utility.th_modeling import M_LSTM, THModelWrapper, ThTrainer
from scripting.utility.torch_utility import th_fix_seed
import numpy as np


class MyModel(THModelWrapper):

    def __init__(
            self,
            vocab_size: int,
            lstm_weights: int = 64,
            answer_units: int = 64,
            l2_regularization: float = 0.
    ):
        super().__init__()
        self.build_model(vocab_size=vocab_size,
                         lstm_weights=lstm_weights,
                         answer_units=answer_units)

        clf_layer_parameters = list(self.model.final_block.named_parameters())
        clf_layer_parameter_names = [f'final_block.{item[0]}' for item in clf_layer_parameters]
        clf_layer_parameters = [item[1] for item in clf_layer_parameters]
        other_parameters = [v for k, v in self.model.named_parameters() if k not in clf_layer_parameter_names]

        self.optimizer = th.optim.Adam(params=[
            {'params': clf_layer_parameters, "lr": 1e-02},
            {'params': other_parameters, "lr": 1e-03}
        ],
            weight_decay=l2_regularization)
        self.criterion = th.nn.CrossEntropyLoss(reduction='mean')

    def build_model(
            self,
            vocab_size: int,
            lstm_weights: int = 64,
            answer_units: int = 64,
    ):
        self.model = M_LSTM(embedding_dimension=50,
                            vocab_size=vocab_size,
                            lstm_weights=lstm_weights,
                            answer_units=answer_units)

    def loss_op(
            self,
            x,
            targets
    ):
        logits = self.model(x)

        # Cross entropy
        ce = self.criterion(logits, targets.long())
        total_loss = ce

        loss_info = dict()
        loss_info['CE'] = ce

        return total_loss, loss_info

    def train_op(
            self,
            x,
            y
    ):
        self.optimizer.zero_grad()

        loss, loss_info = self.loss_op(x=x,
                                       targets=y)
        loss.backward()

        self.optimizer.step()
        return loss, loss_info

    # We need reduce_tracing=True since we have variable size input sequences! (token_ids)
    def batch_fit(
            self,
            x,
            y
    ):
        return self.train_op(x, y)


if __name__ == '__main__':
    # Settings
    samples_amount = 15000
    batch_size = 32
    random_seed = 42
    num_workers = 1
    prefetch = True
    info_basename = 'th_training_example_{}.npy'
    save_info = False
    epochs = 3

    this_path = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(this_path,
                                             os.pardir,
                                             os.pardir,
                                             os.pardir))

    info_save_dir = os.path.join(base_dir,
                                 'scripting',
                                 'lecture_three',
                                 'runtime_modeling')

    log_dir = os.path.join(base_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    Logger.set_log_path(name='logger',
                        log_path=log_dir)
    logger = Logger.get_logger(__name__)

    # Seeding
    fix_seed(seed=random_seed)
    th_fix_seed(seed=random_seed)

    # Loading
    train_df, val_df, test_df = load_ibm2015_dataset(samples_amount=samples_amount)

    # Pre-processing pipeline
    preprocessor = Preprocessor()
    preprocessor.setup(train_df=train_df)

    train_steps = preprocessor.get_steps(data=train_df['Sentence'].values,
                                         batch_size=batch_size)
    train_iterator = partial(preprocessor.make_iterator,
                             df=train_df,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)

    model = MyModel(vocab_size=len(preprocessor.vocab) + 1)
    trainer = ThTrainer(epochs=epochs)
    training_time, memory_usage = trainer.run(model=model,
                                              train_data_iterator=train_iterator,
                                              steps=train_steps)

    logger.info(f'''Times:
        {training_time}
        ''')
    logger.info(f'''Memory usage:
        {memory_usage}
        ''')
    if save_info:
        np.save(os.path.join(info_save_dir, info_basename.format('timing')), training_time)
        np.save(os.path.join(info_save_dir, info_basename.format('memory')), memory_usage)
