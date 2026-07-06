import re

import pandas as pd


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
                'pl_source_row_no': (
                    None if pl_row is None else pl_row.get('source_row_no')
                ),
                'inv_source_row_no': (
                    None if inv_row is None else inv_row.get('source_row_no')
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
                'pl_source_row_no': None,
                'inv_source_row_no': None,
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
