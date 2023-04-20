import os
from functools import partial

import tensorflow as tf

from scripting.lecture_three.dataset_loading.ibm2015_loader import load_ibm2015_dataset
from scripting.lecture_three.dataset_loading.tf_data_pipeline_gen import Preprocessor
from scripting.utility.benchmarking_utility import fix_seed
from scripting.utility.logging_utility import Logger
from scripting.utility.tf_modeling import M_LSTM, TFModelWrapper, TFTrainer
from scripting.utility.tensorflow_utility import tf_fix_seed
import numpy as np


class MyModel(TFModelWrapper):

    def __init__(
            self,
            vocab_size: int,
            lstm_weights: int = 64,
            answer_units: int = 64,
            l2_regularization: float = 0.
    ):
        self.model = M_LSTM(embedding_dimension=50,
                            vocab_size=vocab_size,
                            lstm_weights=lstm_weights,
                            answer_units=answer_units,
                            l2_regularization=l2_regularization)
        self.optimizer = tf.keras.optimizers.Adam()

    def loss_op(
            self,
            x,
            targets,
            training=False
    ):
        logits = self.model(x,
                            training=training)

        # Cross entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                            logits=logits)
        total_loss = tf.reduce_mean(ce)

        loss_info = dict()
        loss_info['CE'] = total_loss

        # L2 regularization
        if self.model.losses:
            additional_losses = tf.reduce_sum(self.model.losses)
            total_loss += additional_losses
            loss_info['L2'] = additional_losses

        return total_loss, loss_info

    def train_op(
            self,
            x,
            y
    ):
        with tf.GradientTape() as tape:
            loss, loss_info = self.loss_op(x=x,
                                           targets=y,
                                           training=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, loss_info, grads

    # We need reduce_tracing=True since we have variable size input sequences! (token_ids)
    @tf.function(reduce_retracing=True)
    def batch_fit(
            self,
            x,
            y
    ):
        loss, loss_info, grads = self.train_op(x, y)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_info


if __name__ == '__main__':
    # Settings
    samples_amount = 15000
    batch_size = 32
    random_seed = 42
    prefetch = True
    info_basename = 'tf_training_gen_example_{}.npy'
    save_info = False
    epochs = 3
    enable_eager_mode = False

    # Enable/Disable eager execution mode
    if enable_eager_mode:
        tf.config.run_functions_eagerly(True)

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
    tf_fix_seed(seed=random_seed)

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
                             prefetch=prefetch,
                             shuffle=True)

    model = MyModel(vocab_size=preprocessor.tokenizer.vocabulary_size() + 1)
    trainer = TFTrainer(epochs=epochs)
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