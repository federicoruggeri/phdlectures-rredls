import time

from memory_profiler import memory_usage
from typing import Callable, List, AnyStr, Iterator
from scripting.utility.logging_utility import Logger
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


def evaluate_time(
        func: Callable
):
    def compute_time(
            *args,
            **kwargs
    ):
        start_time = time.perf_counter()
        func_result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        Logger.get_logger(__name__).info(f'Function {func.__name__} took {total_time} seconds')
        return func_result

    return compute_time


def fix_seed(
        seed: int
):
    Logger.get_logger(__name__).info(f'Fixing seed to: {seed}')
    random.seed(seed)
    np.random.seed(seed)


def _simulate(
        iterator: Callable[[], Iterator],
        steps: int,
        epochs: int = 3,
        simulation_time: float = 0.001,
        description: str = '',
        debug: bool = False
):
    for epoch in range(epochs):
        epoch_iterator = iterator()
        with tqdm(total=steps) as pbar:
            for it_counter, (batch_text, batch_labels) in enumerate(epoch_iterator):
                time.sleep(simulation_time)

                # Debug
                if it_counter == 0 and debug:
                    Logger.get_logger(__name__).info(f'''
                        Batch text: {batch_text} ({batch_text.shape})
                        Batch labels: {batch_labels} ({batch_labels.shape})
                    ''')

                pbar.set_description(desc=f'Simulating {description} -- Epoch {epoch + 1}')
                pbar.update(1)


def simulate_iterator(
        iterator: Callable[[], Iterator],
        steps: int,
        simulation_time: float = 0.001,
        description: str = '',
        debug: bool = False
):
    Logger.get_logger(__name__).info('Simulating iterator consumption...')
    start_time = time.perf_counter()
    mem_usage = memory_usage((_simulate, (), {"iterator": iterator,
                                              "steps": steps,
                                              "simulation_time": simulation_time,
                                              "description": description,
                                              "debug": debug}))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    return total_time, mem_usage


def compare_methods_timing(
        filepaths: List[AnyStr],
        identifiers: List[str],
):
    Logger.get_logger(__name__).info(f'''Comparing methods...
        Filepaths (total={len(filepaths)}): {filepaths}
        Identifiers (total={len(identifiers)}): {identifiers}
        ''')

    assert len(filepaths) == len(identifiers), f'Expected an equal number of methods ({len(filepaths)})' \
                                               f' and identifiers ({len(identifiers)})!'
    filepaths = list(map(lambda p: os.path.normpath(p), filepaths))
    valid_indexes = filter(lambda t: os.path.isfile(t[1]), enumerate(filepaths))
    valid_indexes = list(map(lambda t: t[0], valid_indexes))

    valid_data = [np.load(filepaths[index], allow_pickle=True).item() for index in valid_indexes]
    identifiers = [identifiers[index] for index in valid_indexes]

    data = pd.DataFrame.from_records(data=valid_data)

    fig, ax = plt.subplots(1, 1)
    width = 0.25

    x_labels = ['test', 'val', 'test']
    x_values = np.arange(len(x_labels))
    for row_idx, row in data.iterrows():
        offset = width * row_idx
        rects = ax.bar(x_values + offset, row.values, width)
        ax.bar_label(rects, padding=3)

    ax.set_title('Timing comparison')

    ax.set_ylabel('Timing (secs)')
    ax.set_xlabel('Data split')
    ax.set_xticks(x_values + width / data.shape[0], x_labels)
    ax.legend(identifiers, loc='best')
    plt.show()


def compare_methods_memory_usage(
        filepaths: List[AnyStr],
        identifiers: List[str],
):
    Logger.get_logger(__name__).info(f'''Comparing methods...
        Filepaths (total={len(filepaths)}): {filepaths}
        Identifiers (total={len(identifiers)}): {identifiers}
        ''')

    assert len(filepaths) == len(identifiers), f'Expected an equal number of methods ({len(filepaths)})' \
                                               f' and identifiers ({len(identifiers)})!'
    filepaths = list(map(lambda p: os.path.normpath(p), filepaths))
    valid_indexes = filter(lambda t: os.path.isfile(t[1]), enumerate(filepaths))
    valid_indexes = list(map(lambda t: t[0], valid_indexes))

    valid_data = [np.load(filepaths[index], allow_pickle=True).item() for index in valid_indexes]
    identifiers = [identifiers[index] for index in valid_indexes]

    data = pd.DataFrame.from_records(data=valid_data)
    data_splits = ['train', 'val', 'test']

    fig, axes = plt.subplots(len(data_splits), 1)

    cmap = cm.ScalarMappable(cmap='rainbow')
    for idx, (ax, data_split) in enumerate(zip(axes, data_splits)):
        bp = ax.boxplot(x=data[data_split],
                        patch_artist=True,
                        vert=True,
                        showmeans=True)
        for patch, color in zip(bp['boxes'], cmap.to_rgba(x=[np.mean(seq) for seq in data[data_split]])):
            patch.set_facecolor(color)

        if idx == 0:
            ax.set_title('Memory usage comparison')
            ax.legend(bp['boxes'], identifiers, loc='best')

        ax.set_xlabel('Data Split')
        ax.set_ylabel('Mem Usage (MiB)')

    plt.show()
