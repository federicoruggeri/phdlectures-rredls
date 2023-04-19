import os
from typing import AnyStr

from scripting.utility.benchmarking_utility import compare_methods_timing, compare_methods_memory_usage
from scripting.utility.logging_utility import Logger


def compare_timing(
        info_dir: AnyStr
):
    filepaths = [os.path.join(info_dir, filename) for filename in os.listdir(info_dir)
                 if filename.endswith('timing.npy')]
    identifiers = [os.path.basename(filepath).split('_timing.npy')[0] for filepath in filepaths]

    compare_methods_timing(filepaths=filepaths,
                           identifiers=identifiers)


def compare_memory(
        info_dir: AnyStr
):
    filepaths = [os.path.join(info_dir, filename) for filename in os.listdir(info_dir)
                 if filename.endswith('memory.npy')]
    identifiers = [os.path.basename(filepath).split('_memory.npy')[0] for filepath in filepaths]

    compare_methods_memory_usage(filepaths=filepaths,
                                 identifiers=identifiers)


if __name__ == '__main__':
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

    compare_timing(info_dir=info_save_dir)
    compare_memory(info_dir=info_save_dir)
