from numbers import Number
import re

import pandas as pd


def normalize_text(text: str) -> str:
    text = text.replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


def normalize_column_name(value: object) -> str:
    return re.sub(r'\s+', ' ', str(value).strip()).lower()


def to_number(value, decimal_separator: str = 'auto'):
    if pd.isna(value):
        return None

    if isinstance(value, Number):
        return float(value)

    text = str(value).strip()

    if text == '':
        return None

    try:
        number = parse_number_text(text, decimal_separator=decimal_separator)
    except ValueError:
        return None

    if pd.isna(number):
        return None

    return number


def parse_number_text(text: str, decimal_separator: str = 'auto') -> float:
    text = str(text).strip()

    if decimal_separator == ',':
        text = text.replace('.', '').replace(',', '.')
    elif decimal_separator == '.':
        parts = text.split(',')

        if len(parts) == 2 and '.' not in text and 1 <= len(parts[1]) <= 2:
            text = '.'.join(parts)
        else:
            text = text.replace(',', '')
    elif ',' in text and '.' not in text:
        parts = text.split(',')

        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            text = '.'.join(parts)
        else:
            text = ''.join(parts)
    else:
        text = text.replace(',', '')

    return float(text)


def to_int(value, decimal_separator: str = 'auto'):
    number = to_number(value, decimal_separator=decimal_separator)

    if number is None or pd.isna(number):
        return None

    return int(number)
