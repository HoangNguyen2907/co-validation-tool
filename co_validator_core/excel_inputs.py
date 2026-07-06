import re

import pandas as pd

from .constants import INTERNAL_SOURCE_ROW_NO
from .utils import normalize_column_name, to_int, to_number


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
    source_row_no = pd.Series(df.index + 1, index=df.index)

    df = df.dropna(how='all')
    source_row_no = source_row_no.loc[df.index]
    df.columns = [
        str(col).strip() if pd.notna(col) else f'Unnamed: {index}'
        for index, col in enumerate(header_values)
    ]
    df[INTERNAL_SOURCE_ROW_NO] = source_row_no.values

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


def get_packing_package_context_df(df: pd.DataFrame) -> pd.DataFrame:
    for row_position, (_, row) in enumerate(df.iterrows()):
        values = list(row.values)

        for value in values:
            if normalize_column_name(value) == 'packages':
                start = max(0, row_position - 2)
                end = min(len(df), row_position + 3)
                return df.iloc[start:end].copy()

    return pd.DataFrame(columns=df.columns)


def looks_like_product_code(value) -> bool:
    text = str(value).strip().upper()

    if not re.fullmatch(r'[A-Z0-9]{6,20}', text):
        return False

    return bool(re.search(r'\d', text))


def has_possible_product_context(row) -> bool:
    values = [str(value).strip() for value in row.values if pd.notna(value)]

    if not values:
        return False

    for value in values:
        for token in re.findall(r'\b[A-Z0-9]+\b', value.upper()):
            if looks_like_product_code(token):
                return True

    return False


def is_packing_footer_summary_row(row) -> bool:
    values = [
        normalize_column_name(value)
        for value in row.values
        if pd.notna(value)
    ]
    labels = {
        'total',
        'gross weight',
        'net weight',
        'measurement',
        'packages',
        'package',
    }

    return any(value in labels for value in values)


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
    raw_packing_df = df.copy()

    packing_packages = extract_packing_packages_from_df(
        df,
        decimal_separator=decimal_separator,
    )
    packing_package_context_df = get_packing_package_context_df(df)

    product_col = pick_column(df, [product_code_header])
    qty_col = pick_column(df, [qty_shipped_header])
    net_weight_col = pick_column(df, [net_weight_header])
    item_no_col = pick_packing_item_no_column(df, product_col)

    product_row_mask = df.apply(
        lambda row: is_product_row(
            row,
            product_col=product_col,
            qty_col=qty_col,
            net_weight_col=net_weight_col,
            decimal_separator=decimal_separator,
        ),
        axis=1,
    )
    excluded_df = df[
        ~product_row_mask
        & df.apply(has_possible_product_context, axis=1)
        & ~df.apply(is_packing_footer_summary_row, axis=1)
    ].copy()
    df = df[product_row_mask]

    packing_item_no = (
        df[item_no_col].apply(lambda value: to_int(value))
        if item_no_col is not None
        else pd.Series([None] * len(df), index=df.index)
    )

    result = pd.DataFrame(
        {
            'product_row_no': range(1, len(df) + 1),
            'source_row_no': df[INTERNAL_SOURCE_ROW_NO],
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

    packing_context = {
        'raw_df': raw_packing_df,
        'excluded_rows_df': excluded_df,
        'package_context_df': packing_package_context_df,
    }

    return result.reset_index(drop=True), packing_packages, packing_context


def prepare_invoice_df(
    file,
    product_code_header: str,
    qty_shipped_header: str,
    decimal_separator: str = 'auto',
) -> tuple[pd.DataFrame, dict]:
    df = read_table_from_excel(
        file,
        required_columns=[
            product_code_header,
            qty_shipped_header,
        ],
    )
    raw_invoice_df = df.copy()

    product_col = pick_column(df, [product_code_header])
    qty_col = pick_column(df, [qty_shipped_header])
    item_no_col = pick_packing_item_no_column(df, product_col)

    # 1. Print danh sách các headers (tên cột)
    print("--- HEADERS OF DF ---")
    print(df.columns.tolist())
    print("-" * 30)

    print("--- ĐỐI CHIẾU PRODUCT CODE VÀ QTY SHIPPED ---")
    print(df[[product_col, qty_col]].head(10))
    print("-" * 30)

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

    inv_item_no = (
        df[item_no_col].apply(lambda value: to_int(value))
        if item_no_col is not None
        else pd.Series([None] * len(df), index=df.index)
    )

    result = pd.DataFrame(
        {
            'source_row_no': df[INTERNAL_SOURCE_ROW_NO],
            'item_no': inv_item_no,
            'product_code': df[product_col].astype(str).str.strip(),
            'qty_shipped': df[qty_col].apply(
                lambda value: to_int(
                    value,
                    decimal_separator=decimal_separator,
                )
            ),
        }
    )

    invoice_context = {
        'raw_df': raw_invoice_df,
    }

    return result.reset_index(drop=True), invoice_context


# InvoiceSummaryDataRow represents aggregated invoice information for a single product_code.
InvoiceSummaryDataRow = dict  # keys: product_code, total_qty_shipped, sources (dict mapping filename -> {"item_nos": list, "source_row_nos": list})

# InvoiceMap maps product_code to InvoiceSummaryDataRow
InvoiceMap = dict


def prepare_invoice_df_multi(
    files: list[tuple[str, object]],
    product_code_header: str,
    qty_shipped_header: str,
    decimal_separator: str = 'auto',
) -> tuple[InvoiceMap, list[dict]]:
    """Prepare an InvoiceMap and list of invoice contexts from multiple files.

    Args:
        files: list of (filename, file_obj) tuples.
        product_code_header: column header for product code.
        qty_shipped_header: column header for qty shipped.
        decimal_separator: decimal separator option.

    Returns:
        A tuple of:
          - InvoiceMap: dict mapping product_code (upper-stripped) to a single
            InvoiceSummaryDataRow dict.
          - invoice_contexts: list of per-file context dicts (with 'raw_df' and
            'source_file' keys), one per uploaded file.
    """
    invoice_map: InvoiceMap = {}
    invoice_contexts: list[dict] = []

    for filename, file_obj in files:
        print('filename', filename)
        inv_df, ctx = prepare_invoice_df(
            file_obj,
            product_code_header=product_code_header,
            qty_shipped_header=qty_shipped_header,
            decimal_separator=decimal_separator,
        )
        ctx['source_file'] = filename
        invoice_contexts.append(ctx)

        for _, row in inv_df.iterrows():
            product_code = str(row['product_code']).strip().upper()
            qty = row['qty_shipped']
            item_no = row.get('item_no')
            row_no = row['source_row_no']

            if product_code not in invoice_map:
                invoice_map[product_code] = {
                    'product_code': product_code,
                    'total_qty_shipped': 0,
                    'sources': {}
                }

            summary = invoice_map[product_code]
            if qty is not None:
                summary['total_qty_shipped'] += qty

            sources = summary['sources']
            if filename not in sources:
                sources[filename] = {
                    'item_nos': [],
                    'source_row_nos': []
                }

            file_data = sources[filename]
            if item_no is not None and pd.notna(item_no):
                file_data['item_nos'].append(int(item_no))
            if row_no is not None and pd.notna(row_no):
                file_data['source_row_nos'].append(int(row_no))

    return invoice_map, invoice_contexts
