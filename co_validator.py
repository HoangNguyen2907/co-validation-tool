import io
from numbers import Number
import re
import tempfile
from pathlib import Path

import fitz
import pandas as pd
import pdfplumber
import pytesseract
import streamlit as st
from PIL import Image


ITEM_START_RE = re.compile(
    r'(?m)(?:^|\n)\s*(\S{1,3})\s*(?:[|/]\s*)?N/M\s+'
)

HS_RE = re.compile(r'HS\s*Code\s*:\s*(\d+)', re.I)

QTY_RE = re.compile(r'(\d+(?:,\d+)?)\s*PIECES', re.I)

WEIGHT_RE = re.compile(
    r'N\s*\.?\s*W\s*E\s*I\s*G\s*H\s*T\s*[:\s\n]*([\d,.]+)\s*KGS?\b',
    re.I,
)


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
        return parse_number_text(text, decimal_separator=decimal_separator)
    except ValueError:
        return None


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

    if number is None:
        return None

    return int(number)


def repair_item_no(raw_item_no, previous_item_no, next_raw_item_no=None):
    expected = previous_item_no + 1 if previous_item_no is not None else None

    if raw_item_no.isdigit():
        current = int(raw_item_no)

        if expected is None:
            return current, False

        if current == expected:
            return current, False

        if str(expected).endswith(str(current)):
            return expected, True

        if next_raw_item_no and next_raw_item_no.isdigit():
            if int(next_raw_item_no) == expected + 1:
                return expected, True

        return current, False

    if expected is not None:
        if next_raw_item_no and next_raw_item_no.isdigit():
            if int(next_raw_item_no) == expected + 1:
                return expected, True

        return expected, True

    return None, False


def iter_item_blocks(text: str):
    matches = list(ITEM_START_RE.finditer(text))
    previous_item_no = None

    for index, match in enumerate(matches):
        raw_item_no = match.group(1)

        next_raw_item_no = (
            matches[index + 1].group(1)
            if index + 1 < len(matches)
            else None
        )

        item_no, repaired = repair_item_no(
            raw_item_no,
            previous_item_no,
            next_raw_item_no,
        )

        if item_no is None:
            continue

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = normalize_text(text[start:end])

        yield item_no, block, raw_item_no, repaired

        previous_item_no = item_no


def extract_text_pdfplumber(pdf_path: str) -> str:
    parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or '')

    return normalize_text('\n'.join(parts))


def extract_text_ocr(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []

    for page in doc:
        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes('png')))

        text = pytesseract.image_to_string(
            img,
            lang='eng',
            config='--psm 6',
        )

        parts.append(text)

    return normalize_text('\n'.join(parts))


def should_use_ocr(text: str) -> bool:
    if len(text.strip()) < 500:
        return True

    if 'HS Code' not in text and 'HS CODE' not in text:
        return True

    return False


def get_pdf_text(pdf_path: str) -> str:
    text = extract_text_pdfplumber(pdf_path)

    if should_use_ocr(text):
        text = extract_text_ocr(pdf_path)

    return text


def extract_weight(block: str, decimal_separator: str = 'auto'):
    match = WEIGHT_RE.search(block)

    if match:
        return parse_number_text(match.group(1), decimal_separator=decimal_separator)

    weight_pos = re.search(r'WEIGHT', block, re.I)

    if weight_pos:
        tail = block[weight_pos.end():]
        match = re.search(r'([\d,.]+)\s*KGS?\b', tail, re.I)

        if match:
            return parse_number_text(match.group(1), decimal_separator=decimal_separator)

    return None


def clean_description(block: str) -> str:
    desc = HS_RE.split(block)[0]

    desc = re.sub(
        r'\b\d{2}[A-Z]{2,}\d+\b.*$',
        '',
        desc,
        flags=re.I,
    )

    desc = re.sub(
        r'\b(WO|PE|CTH|PSR|\d+%)\b\s+\d+(?:,\d+)?\s*PIECES.*$',
        '',
        desc,
        flags=re.I,
    )

    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip()


def extract_co_items(text: str, decimal_separator: str = 'auto') -> pd.DataFrame:
    rows = []

    text = re.sub(
        r'1\. Parties which accept this form.*?(?=Attachment|Original|Page \d+ of \d+|$)',
        '',
        text,
        flags=re.S,
    )

    for item_no, block, raw_item_no, repaired in iter_item_blocks(text):
        hs = HS_RE.search(block)
        qty = QTY_RE.search(block)

        rows.append(
            {
                'item_no': item_no,
                'raw_item_no': raw_item_no,
                'item_no_repaired': repaired,
                'raw_block': block,
                'description': clean_description(block),
                'hs_code': hs.group(1) if hs else '',
                'quantity_pieces': (
                    int(qty.group(1).replace(',', ''))
                    if qty
                    else None
                ),
                'net_weight_kgs': extract_weight(
                    block,
                    decimal_separator=decimal_separator,
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = (
            df
            .drop_duplicates(subset=['item_no'])
            .sort_values('item_no')
            .reset_index(drop=True)
        )

    return df


def extract_co_total_packages(text: str):
    matches = re.findall(
        r'\bTOTAL\b[\s\S]{0,300}?\((\d+)\)',
        text,
        flags=re.I,
    )

    if not matches:
        return None

    return int(matches[-1])


def read_table_from_excel(file, required_columns: list[str]) -> pd.DataFrame:
    file.seek(0)
    raw = pd.read_excel(file, header=None)

    required = {
        normalize_column_name(col)
        for col in required_columns
    }

    header_index = None

    for index, row in raw.iterrows():
        values = {
            normalize_column_name(value)
            for value in row
            if pd.notna(value)
        }

        if required.issubset(values):
            header_index = index
            break

    if header_index is None:
        raise ValueError(f'Cannot find header row: {required_columns}')

    header_values = raw.iloc[header_index]
    df = raw.iloc[header_index + 1:].copy()

    df = df.dropna(how='all')
    df.columns = [
        str(col).strip() if pd.notna(col) else f'Unnamed: {index}'
        for index, col in enumerate(header_values)
    ]

    return df


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized_map = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in normalized_map:
            return normalized_map[key]

    raise ValueError(f'Missing column. Candidates: {candidates}')


def pick_packing_item_no_column(df: pd.DataFrame, product_col: str) -> str | None:
    product_index = list(df.columns).index(product_col)
    candidate_columns = list(df.columns[:product_index])

    for column in candidate_columns:
        numeric_values = df[column].apply(to_number).dropna()

        if numeric_values.empty:
            continue

        if numeric_values.apply(float.is_integer).all():
            return column

    return None


def is_product_row(
    row,
    product_col: str,
    qty_col: str,
    net_weight_col: str | None = None,
    decimal_separator: str = 'auto',
) -> bool:
    product_code = str(row.get(product_col, '')).strip()
    raw_qty_col = row.get(qty_col)
    qty = to_number(raw_qty_col, decimal_separator=decimal_separator)
    if (
        product_code == ''
        or product_code.lower() == 'nan'
        or qty is None
    ):
        return False

    if not re.fullmatch(r'[A-Z0-9]{6,20}', product_code.upper()):
        return False

    if net_weight_col is not None:
        net_weight = to_number(
            row.get(net_weight_col),
            decimal_separator=decimal_separator,
        )

        if net_weight is None:
            return False

    return True


def extract_packing_packages_from_df(
    df: pd.DataFrame,
    decimal_separator: str = 'auto',
):
    for _, row in df.iterrows():
        values = list(row.values)

        for index, value in enumerate(values):
            if normalize_column_name(value) == 'packages':
                for next_value in values[index + 1:]:
                    number = to_number(
                        next_value,
                        decimal_separator=decimal_separator,
                    )

                    if number is not None:
                        return int(number)

    return None


def prepare_packing_df(
    file,
    product_code_header: str,
    qty_shipped_header: str,
    net_weight_header: str,
    decimal_separator: str = 'auto',
):
    df = read_table_from_excel(
        file,
        required_columns=[
            product_code_header,
            qty_shipped_header,
            net_weight_header,
        ],
    )

    packing_packages = extract_packing_packages_from_df(
        df,
        decimal_separator=decimal_separator,
    )

    product_col = pick_column(df, [product_code_header])
    qty_col = pick_column(df, [qty_shipped_header])
    net_weight_col = pick_column(df, [net_weight_header])
    item_no_col = pick_packing_item_no_column(df, product_col)

    df = df[
        df.apply(
            lambda row: is_product_row(
                row,
                product_col=product_col,
                qty_col=qty_col,
                net_weight_col=net_weight_col,
                decimal_separator=decimal_separator,
            ),
            axis=1,
        )
    ]

    packing_item_no = (
        df[item_no_col].apply(lambda value: to_int(value))
        if item_no_col is not None
        else pd.Series([None] * len(df), index=df.index)
    )

    result = pd.DataFrame(
        {
            'product_row_no': range(1, len(df) + 1),
            'source_row_no': df.index + 1,
            'packing_list_item_no': packing_item_no,
            'product_code': df[product_col].astype(str).str.strip(),
            'qty_shipped': df[qty_col].apply(
                lambda value: to_int(
                    value,
                    decimal_separator=decimal_separator,
                )
            ),
            'net_weight': df[net_weight_col].apply(
                lambda value: to_number(
                    value,
                    decimal_separator=decimal_separator,
                )
            ),
        }
    )

    return result.reset_index(drop=True), packing_packages


def prepare_invoice_df(
    file,
    product_code_header: str,
    qty_shipped_header: str,
    decimal_separator: str = 'auto',
) -> pd.DataFrame:
    df = read_table_from_excel(
        file,
        required_columns=[
            product_code_header,
            qty_shipped_header,
        ],
    )

    product_col = pick_column(df, [product_code_header])
    qty_col = pick_column(df, [qty_shipped_header])

    df = df[
        df.apply(
            lambda row: is_product_row(
                row,
                product_col=product_col,
                qty_col=qty_col,
                decimal_separator=decimal_separator,
            ),
            axis=1,
        )
    ]

    result = pd.DataFrame(
        {
            'source_row_no': df.index + 1,
            'product_code': df[product_col].astype(str).str.strip(),
            'qty_shipped': df[qty_col].apply(
                lambda value: to_int(
                    value,
                    decimal_separator=decimal_separator,
                )
            ),
        }
    )

    return result.reset_index(drop=True)


def compare_pl_invoice(pl_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    len_pl = len(pl_df)
    len_inv = len(inv_df)

    max_len = max(len_pl, len_inv)

    for index in range(max_len):
        pl_row = pl_df.iloc[index] if index < len_pl else None
        inv_row = inv_df.iloc[index] if index < len_inv else None

        if pl_row is None:
            status = 'ROW_MISSING_IN_PL'
            note = 'Invoice has row but Packing List does not.'
        elif inv_row is None:
            status = 'ROW_MISSING_IN_INVOICE'
            note = 'Packing List has row but Invoice does not.'
        else:
            product_match = pl_row['product_code'] == inv_row['product_code']
            qty_match = pl_row['qty_shipped'] == inv_row['qty_shipped']

            if product_match and qty_match:
                status = 'PASS'
                note = ''
            elif not product_match and not qty_match:
                status = 'PRODUCT_AND_QTY_MISMATCH'
                note = 'Product Code and Qty Shipped are different.'
            elif not product_match:
                status = 'PRODUCT_MISMATCH'
                note = 'Product Code is different.'
            else:
                status = 'QTY_MISMATCH'
                note = 'Qty Shipped is different.'

        rows.append(
            {
                'row_no': index + 1,
                'packing_list_item_no': (
                    None
                    if pl_row is None
                    else pl_row.get('packing_list_item_no')
                ),
                'pl_product_code': None if pl_row is None else pl_row['product_code'],
                'inv_product_code': None if inv_row is None else inv_row['product_code'],
                'pl_qty_shipped': None if pl_row is None else pl_row['qty_shipped'],
                'inv_qty_shipped': None if inv_row is None else inv_row['qty_shipped'],
                'status': status,
                'note': note,
            }
        )

    return pd.DataFrame(rows)


def build_pl_invoice_not_run_df(note: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'row_no': None,
                'packing_list_item_no': None,
                'pl_product_code': None,
                'inv_product_code': None,
                'pl_qty_shipped': None,
                'inv_qty_shipped': None,
                'status': 'NOT_RUN',
                'note': note,
            }
        ]
    )


def build_packing_summary(pl_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pl_df
        .groupby('product_code', as_index=False)
        .agg(
            total_shipped_qty=('qty_shipped', 'sum'),
            total_net_weight=('net_weight', 'sum'),
            row_count=('product_code', 'size'),
            product_row_nos=(
                'product_row_no',
                lambda values: ', '.join(map(str, values)),
            ),
            packing_list_item_nos=(
                'packing_list_item_no',
                lambda values: ', '.join(
                    str(int(value)) if pd.notna(value) else ''
                    for value in values
                ),
            ),
            excel_file_row_nos=(
                'source_row_no',
                lambda values: ', '.join(map(str, values)),
            ),
        )
    )


def tokenize_description(description: str) -> set[str]:
    return set(
        re.findall(
            r'\b[A-Z0-9]+\b',
            str(description).upper(),
        )
    )


def build_co_token_index(co_df: pd.DataFrame) -> dict[str, list[dict]]:
    token_to_co_rows: dict[str, list[dict]] = {}

    for _, row in co_df.iterrows():
        row_dict = row.to_dict()
        tokens = tokenize_description(row_dict.get('description', ''))

        for token in tokens:
            token_to_co_rows.setdefault(token, []).append(row_dict)

    return token_to_co_rows


def validate_packing_vs_co(
    packing_summary_df: pd.DataFrame,
    co_df: pd.DataFrame,
    weight_tolerance: float,
) -> pd.DataFrame:
    co_index = build_co_token_index(co_df)
    rows = []

    for _, packing_row in packing_summary_df.iterrows():
        product_code = str(packing_row['product_code']).upper().strip()
        matched_co_rows = co_index.get(product_code, [])

        if len(matched_co_rows) == 0:
            rows.append(
                {
                    'product_code': product_code,
                    'co_item_no': None,
                    'product_row_no': packing_row['product_row_nos'],
                    'packing_list_item_no': packing_row['packing_list_item_nos'],
                    'packing_total_qty': packing_row['total_shipped_qty'],
                    'co_quantity_pieces': None,
                    'qty_diff': None,
                    'packing_total_net_weight': packing_row['total_net_weight'],
                    'co_net_weight_kgs': None,
                    'weight_diff': None,
                    'co_description': None,
                    'status': 'NOT_FOUND_IN_CO',
                    'note': 'Product Code from Packing List was not found in CO.',
                }
            )
            continue

        if len(matched_co_rows) > 1:
            rows.append(
                {
                    'product_code': product_code,
                    'co_item_no': ', '.join(
                        str(row.get('item_no', ''))
                        for row in matched_co_rows
                    ),
                    'product_row_no': packing_row['product_row_nos'],
                    'packing_list_item_no': packing_row['packing_list_item_nos'],
                    'packing_total_qty': packing_row['total_shipped_qty'],
                    'co_quantity_pieces': None,
                    'qty_diff': None,
                    'packing_total_net_weight': packing_row['total_net_weight'],
                    'co_net_weight_kgs': None,
                    'weight_diff': None,
                    'co_description': ' | '.join(
                        str(row.get('description', ''))
                        for row in matched_co_rows
                    ),
                    'status': 'DUPLICATE_IN_CO',
                    'note': 'Product Code appears in more than one CO row.',
                }
            )
            continue

        co_row = matched_co_rows[0]
        co_qty = co_row.get('quantity_pieces')
        co_weight = co_row.get('net_weight_kgs')

        if co_qty is None or co_weight is None:
            rows.append(
                {
                    'product_code': product_code,
                    'co_item_no': co_row.get('item_no'),
                    'product_row_no': packing_row['product_row_nos'],
                    'packing_list_item_no': packing_row['packing_list_item_nos'],
                    'packing_total_qty': packing_row['total_shipped_qty'],
                    'co_quantity_pieces': co_qty,
                    'qty_diff': None,
                    'packing_total_net_weight': packing_row['total_net_weight'],
                    'co_net_weight_kgs': co_weight,
                    'weight_diff': None,
                    'co_description': co_row.get('description'),
                    'status': 'CO_VALUE_MISSING',
                    'note': 'CO quantity or net weight is missing.',
                }
            )
            continue

        qty_diff = packing_row['total_shipped_qty'] - co_qty
        weight_diff = packing_row['total_net_weight'] - co_weight

        qty_match = qty_diff == 0
        weight_match = abs(weight_diff) <= weight_tolerance

        if qty_match and weight_match:
            status = 'PASS'
            note = ''
        elif not qty_match and not weight_match:
            status = 'QTY_AND_WEIGHT_MISMATCH'
            note = 'Quantity and Net Weight are different.'
        elif not qty_match:
            status = 'QTY_MISMATCH'
            note = 'Quantity is different.'
        else:
            status = 'WEIGHT_MISMATCH'
            note = 'Net Weight is different.'

        rows.append(
            {
                'product_code': product_code,
                'co_item_no': co_row.get('item_no'),
                'product_row_no': packing_row['product_row_nos'],
                'packing_list_item_no': packing_row['packing_list_item_nos'],
                'packing_total_qty': packing_row['total_shipped_qty'],
                'co_quantity_pieces': co_qty,
                'qty_diff': qty_diff,
                'packing_total_net_weight': packing_row['total_net_weight'],
                'co_net_weight_kgs': co_weight,
                'weight_diff': weight_diff,
                'co_description': co_row.get('description'),
                'status': status,
                'note': note,
            }
        )

    return pd.DataFrame(rows)


def build_packing_vs_co_not_run_df(note: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'product_code': None,
                'co_item_no': None,
                'product_row_no': None,
                'packing_list_item_no': None,
                'packing_total_qty': None,
                'co_quantity_pieces': None,
                'qty_diff': None,
                'packing_total_net_weight': None,
                'co_net_weight_kgs': None,
                'weight_diff': None,
                'co_description': None,
                'status': 'NOT_RUN',
                'note': note,
            }
        ]
    )


def build_package_check_df(packing_packages, co_total_packages) -> pd.DataFrame:
    if packing_packages is None:
        status = 'PACKING_PACKAGES_NOT_FOUND'
        note = 'Cannot find Packages in Packing List footer.'
    elif co_total_packages is None:
        status = 'CO_TOTAL_PACKAGES_NOT_FOUND'
        note = 'Cannot find TOTAL packages in CO.'
    elif packing_packages == co_total_packages:
        status = 'PASS'
        note = ''
    else:
        status = 'PACKAGE_MISMATCH'
        note = 'Packing List Packages is different from CO TOTAL packages.'

    return pd.DataFrame(
        [
            {
                'packing_packages': packing_packages,
                'co_total_packages': co_total_packages,
                'status': status,
                'note': note,
            }
        ]
    )


def build_package_check_not_run_df(note: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'packing_packages': None,
                'co_total_packages': None,
                'status': 'NOT_RUN',
                'note': note,
            }
        ]
    )


def build_empty_co_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'item_no',
            'raw_item_no',
            'item_no_repaired',
            'raw_block',
            'description',
            'hs_code',
            'quantity_pieces',
            'net_weight_kgs',
        ]
    )


def count_validation_errors(df: pd.DataFrame) -> int:
    if df.empty or 'status' not in df.columns:
        return 0

    return int((~df['status'].isin(['PASS', 'NOT_RUN'])).sum())


def build_summary(
    pl_df,
    inv_df,
    co_df,
    pl_vs_invoice_df,
    packing_vs_co_df,
    package_check_df,
    missing_files: list[str] | None = None,
):
    missing_files = missing_files or []

    pl_invoice_errors = count_validation_errors(pl_vs_invoice_df)
    packing_co_errors = count_validation_errors(packing_vs_co_df)
    package_errors = count_validation_errors(package_check_df)

    final_result = (
        'PASS'
        if pl_invoice_errors == 0
        and packing_co_errors == 0
        and package_errors == 0
        else 'FAIL'
    )

    missing_file_notes = {
        'Packing List': 'No Packing List available.',
        'Invoice': 'No Invoice available.',
        'CO': 'No CO available.',
    }

    def note_for(file_name: str) -> str:
        return missing_file_notes[file_name] if file_name in missing_files else ''

    rows = [
        {
            'metric': 'Packing rows',
            'value': str(len(pl_df)),
            'note': note_for('Packing List'),
        },
        {
            'metric': 'Invoice rows',
            'value': str(len(inv_df)),
            'note': note_for('Invoice'),
        },
        {
            'metric': 'CO rows',
            'value': str(len(co_df)),
            'note': note_for('CO'),
        },
        {
            'metric': 'PL vs Invoice errors',
            'value': str(pl_invoice_errors),
            'note': (
                'Not run because Invoice is not available.'
                if 'Invoice' in missing_files
                else ''
            ),
        },
        {
            'metric': 'Packing vs CO errors',
            'value': str(packing_co_errors),
            'note': (
                'Not run because CO is not available.'
                if 'CO' in missing_files
                else ''
            ),
        },
        {
            'metric': 'Package check errors',
            'value': str(package_errors),
            'note': (
                'Not run because CO is not available.'
                if 'CO' in missing_files
                else ''
            ),
        },
        {'metric': 'Final result', 'value': final_result, 'note': ''},
    ]

    return pd.DataFrame(rows)


def write_report(
    output_path,
    summary_df,
    pl_vs_invoice_df,
    packing_summary_df,
    packing_vs_co_df,
    package_check_df,
    co_df,
):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        pl_vs_invoice_df.to_excel(writer, sheet_name='PL_vs_Invoice', index=False)
        packing_summary_df.to_excel(writer, sheet_name='Packing_Summary', index=False)
        packing_vs_co_df.to_excel(writer, sheet_name='Packing_vs_CO', index=False)
        package_check_df.to_excel(writer, sheet_name='Package_Check', index=False)
        co_df.to_excel(writer, sheet_name='CO', index=False)


st.set_page_config(
    page_title='CO Validation Tool',
    layout='wide',
)

st.title('CO Validation Tool')

DECIMAL_SEPARATOR_OPTIONS = {
    'Comma (,)': ',',
    'Dot (.)': '.',
    'Auto detect': 'auto',
}

st.subheader('Packing List Columns')

packing_file = st.file_uploader(
    'Upload Packing List Excel',
    type=['xlsx'],
)

pl_product_code_header = st.text_input(
    'Packing List Product Code header',
    value='Product Code',
)

pl_qty_shipped_header = st.text_input(
    'Packing List Qty Shipped header',
    value='Qty Shipped',
)

pl_net_weight_header = st.text_input(
    'Packing List Net Weight header',
    value='Net Weight',
)

pl_decimal_separator_label = st.selectbox(
    'Packing List decimal separator',
    options=list(DECIMAL_SEPARATOR_OPTIONS.keys()),
    index=0,
)

st.subheader('Commercial Invoice Columns')

invoice_file = st.file_uploader(
    'Upload Commercial Invoice Excel',
    type=['xlsx'],
)

inv_product_code_header = st.text_input(
    'Invoice Product Code header',
    value='Product Code',
)

inv_qty_shipped_header = st.text_input(
    'Invoice Qty Shipped header',
    value='Qty Shipped',
)

inv_decimal_separator_label = st.selectbox(
    'Invoice decimal separator',
    options=list(DECIMAL_SEPARATOR_OPTIONS.keys()),
    index=0,
)

st.subheader('CO Document')

co_pdf_file = st.file_uploader(
    'Upload CO PDF',
    type=['pdf'],
)

co_decimal_separator_label = st.selectbox(
    'CO decimal separator',
    options=list(DECIMAL_SEPARATOR_OPTIONS.keys()),
    index=1,
)

weight_tolerance = st.number_input(
    'Weight tolerance',
    value=0.01,
    step=0.01,
)

files_ready = packing_file is not None and (
    invoice_file is not None or co_pdf_file is not None
)

validate_clicked = st.button(
    'Validate',
    type='primary',
    disabled=not files_ready,
)

if not files_ready:
    st.info(
        'Upload the Packing List and at least one comparison file: '
        'Commercial Invoice or CO PDF.'
    )

if validate_clicked:
    with st.spinner('Reading files and validating...'):
        pl_decimal_separator = DECIMAL_SEPARATOR_OPTIONS[pl_decimal_separator_label]
        inv_decimal_separator = DECIMAL_SEPARATOR_OPTIONS[inv_decimal_separator_label]
        co_decimal_separator = DECIMAL_SEPARATOR_OPTIONS[co_decimal_separator_label]

        pl_df, packing_packages = prepare_packing_df(
            packing_file,
            product_code_header=pl_product_code_header,
            qty_shipped_header=pl_qty_shipped_header,
            net_weight_header=pl_net_weight_header,
            decimal_separator=pl_decimal_separator,
        )

        missing_files = []

        if invoice_file is None:
            missing_files.append('Invoice')
            inv_df = pd.DataFrame(
                columns=[
                    'source_row_no',
                    'product_code',
                    'qty_shipped',
                ]
            )
            pl_vs_invoice_df = build_pl_invoice_not_run_df(
                'No Invoice available. PL vs Invoice validation was not run.'
            )
        else:
            inv_df = prepare_invoice_df(
                invoice_file,
                product_code_header=inv_product_code_header,
                qty_shipped_header=inv_qty_shipped_header,
                decimal_separator=inv_decimal_separator,
            )
            pl_vs_invoice_df = compare_pl_invoice(pl_df, inv_df)

        if co_pdf_file is None:
            missing_files.append('CO')
            co_text = ''
            co_df = build_empty_co_df()
            co_total_packages = None
            packing_vs_co_df = build_packing_vs_co_not_run_df(
                'No CO available. Packing vs CO validation was not run.'
            )
            package_check_df = build_package_check_not_run_df(
                'No CO available. Package check was not run.'
            )
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(co_pdf_file.read())
                co_pdf_path = tmp.name

            co_text = get_pdf_text(co_pdf_path)
            co_df = extract_co_items(
                co_text,
                decimal_separator=co_decimal_separator,
            )
            co_total_packages = extract_co_total_packages(co_text)

        packing_summary_df = build_packing_summary(pl_df)

        if co_pdf_file is not None:
            packing_vs_co_df = validate_packing_vs_co(
                packing_summary_df,
                co_df,
                weight_tolerance=weight_tolerance,
            )

            package_check_df = build_package_check_df(
                packing_packages,
                co_total_packages,
            )

        summary_df = build_summary(
            pl_df,
            inv_df,
            co_df,
            pl_vs_invoice_df,
            packing_vs_co_df,
            package_check_df,
            missing_files=missing_files,
        )

        output_path = Path(tempfile.gettempdir()) / 'co_validation_result.xlsx'

        write_report(
            output_path,
            summary_df,
            pl_vs_invoice_df,
            packing_summary_df,
            packing_vs_co_df,
            package_check_df,
            co_df,
        )

    final_result = summary_df.loc[
        summary_df['metric'] == 'Final result',
        'value',
    ].iloc[0]

    if final_result == 'PASS':
        st.success('Validation passed')
    else:
        st.error('Validation failed') 
    st.subheader('Summary')
    st.dataframe(summary_df, hide_index=True, width='stretch')

    st.subheader('PL vs Invoice')
    st.dataframe(pl_vs_invoice_df, hide_index=True, width='stretch')

    st.subheader('Packing Summary')
    st.dataframe(packing_summary_df, hide_index=True, width='stretch')

    st.subheader('Packing vs CO')
    st.dataframe(packing_vs_co_df, hide_index=True, width='stretch')

    st.subheader('Package Check')
    st.dataframe(package_check_df, hide_index=True, width='stretch')

    st.subheader('CO')
    st.dataframe(co_df, hide_index=True, width='stretch')

    # st.download_button(
    #     label='Download validation report',
    #     data=output_path.read_bytes(),
    #     file_name='co_validation_result.xlsx',
    #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    # )

    # with st.expander('Preview CO extracted text'):
    #     st.text(co_text)
