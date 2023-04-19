import os
from typing import AnyStr

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripting.utility.logging_utility import Logger
from scripting.utility.benchmarking_utility import evaluate_time


class IBM2015Loader:

    def __init__(
            self,
            load_path: AnyStr
    ):
        self.load_path = load_path

    def load(
            self
    ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.load_path)
        train_test_df = df[df['Data-set'] == 'train and test']
        val_df = df[df['Data-set'] == 'held-out']

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
        train_indexes, test_indexes = list(splitter.split(X=train_test_df['Sentence'].values,
                                                          y=train_test_df['Label'].values,
                                                          groups=train_test_df['Topic id'].values))[0]
        train_df = train_test_df.iloc[train_indexes]
        test_df = train_test_df.iloc[test_indexes]

        return train_df, val_df, test_df


@evaluate_time
def load_ibm2015_dataset(
        samples_amount: int = -1
):
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

    load_path = os.path.normpath(os.path.join(base_dir,
                                              'data',
                                              'lecture_three',
                                              'dataset.csv'))
    logger.info(f'Attempting to load dataset from path: {load_path}')

    loader = IBM2015Loader(load_path=load_path)
    train_df, val_df, test_df = loader.load()

    if samples_amount > 0:
        logger.info(f'Samples amount given: {samples_amount} -- Taking a slice of retrieved datasets...')
        train_df = train_df[:samples_amount]
        val_df = val_df[:samples_amount]
        test_df = test_df[:samples_amount]

    logger.info(f'''Loaded data: 
                Train: {train_df.shape}
                Val: {val_df.shape}
                Test: {test_df.shape}
                ''')
    return train_df, val_df, test_df


if __name__ == '__main__':
    load_ibm2015_dataset()
