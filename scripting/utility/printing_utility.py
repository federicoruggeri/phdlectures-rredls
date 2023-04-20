"""

Mainly inspired from how ParlAI shows results

"""

import re

import numpy as np
import pandas as pd

LINE_WIDTH = 700


def float_formatter(f) -> str:
    """
    Format a float as a pretty string.
    """
    if f != f or f is None:
        # instead of returning nan, return "" so it shows blank in table
        return ""
    if isinstance(f, int):
        # don't do any rounding of integers, leave them alone
        return str(f)
    if f >= 1000:
        # numbers > 1000 just round to the nearest integer
        s = f'{f:.0f}'
    else:
        # otherwise show 4 significant figures, regardless of decimal spot
        s = f'{f:.4g}'
    # replace leading 0's with blanks for easier reading
    # example:  -0.32 to -.32
    s = s.replace('-0.', '-.')
    if s.startswith('0.'):
        s = s[1:]
    # Add the trailing 0's to always show 4 digits
    # example: .32 to .3200
    if s[0] == '.' and len(s) < 5:
        s += '0' * (5 - len(s))
    return s


def general_formatter(value):
    if isinstance(value, list) or isinstance(value, np.ndarray):
        # apply formatting to each element
        return '[{}]'.format(','.join([float_formatter(item) for item in value]))
    if isinstance(value, dict):
        return '[{}]'.format(','.join(list(value.items())))


def prettify_value(value):
    if value is None:
        return ""
    elif type(value) in [np.ndarray, list, dict]:
        return general_formatter(value)
    elif type(value) != str:
        try:
            return float_formatter(value)
        except:
            return value
    else:
        return value


def prettify_statistics(statistics, ignore_non_floats=False):
    non_float_columns = [column for column, value in statistics.items() if type(value) in [np.ndarray, list, dict]]
    df = pd.DataFrame([statistics])

    if ignore_non_floats:
        df.drop(non_float_columns, inplace=True)

    result = "   " + df.to_string(
        na_rep="",
        line_width=LINE_WIDTH - 3,  # -3 for the extra spaces we add
        float_format=float_formatter,
        formatters={column: general_formatter for column in non_float_columns},
        index=df.shape[0] > 1,
    ).replace("\n\n", "\n").replace("\n", "\n   ")
    result = re.sub(r"\s+$", "", result)

    return result
