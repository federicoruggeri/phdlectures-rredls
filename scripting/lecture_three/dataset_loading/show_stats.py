import os

from scripting.utility.benchmarking_utility import compare_methods_timing, compare_methods_memory_usage
from scripting.utility.logging_utility import Logger


def compare_timing():
    filepaths = [os.path.join(this_path, filename) for filename in os.listdir(this_path)
                 if filename.endswith('timing.npy')]
    identifiers = [os.path.basename(filepath).split('timing.npy')[0] for filepath in filepaths]

    compare_methods_timing(filepaths=filepaths,
                           identifiers=identifiers)


def compare_memory():
    filepaths = [os.path.join(this_path, filename) for filename in os.listdir(this_path)
                 if filename.endswith('memory.npy')]
    identifiers = [os.path.basename(filepath).split('memory.npy')[0] for filepath in filepaths]

    compare_methods_memory_usage(filepaths=filepaths,
                                 identifiers=identifiers)


if __name__ == '__main__':
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

    compare_timing()
    compare_memory()
