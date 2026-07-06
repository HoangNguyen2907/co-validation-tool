import re
from copy import deepcopy

import pandas as pd


def format_grouped_sources(sources: dict, key: str) -> str:
    """Format item numbers or row numbers grouped by source file.
    
    Example: "33, 34, 35 (invoice_1.xlsx); 80 (invoice_2.xlsx)"
    """
    if not sources:
        return ''
    parts = []
    for filename in sorted(sources.keys()):
        data = sources[filename]
        vals = data.get(key, [])
        clean_vals = sorted(list({str(v) for v in vals if v is not None}))
        if clean_vals:
            vals_str = ', '.join(clean_vals)
            if len(sources) > 1:
                parts.append(f"{vals_str} ({filename})")
            else:
                parts.append(vals_str)
    return '; '.join(parts)


def compare_pl_invoice_summary(
    pl_df: pd.DataFrame,
    invoice_map: dict,
) -> pd.DataFrame:
    """Compare Packing List summary against Invoice summary map by product_code.

    - Summarizes both PL and Invoice data by product_code.
    - Compares pl_total_qty vs inv_total_qty.
    - Formats item_nos, source_files, and source_row_nos.
    """
    invoice_map = deepcopy(invoice_map)
    
    # 1. Group Packing List data by product_code
    pkl_grouped = {}
    for _, row in pl_df.iterrows():
        prod_code = str(row['product_code']).strip().upper()
        pkl_grouped.setdefault(prod_code, []).append(row)

    pkl_summaries = {}
    for prod_code, rows in pkl_grouped.items():
        total_qty = sum(r['qty_shipped'] for r in rows if r['qty_shipped'] is not None)
        total_net_w = sum(r['net_weight'] for r in rows if r['net_weight'] is not None)
        
        item_nos = ', '.join(sorted(list({
            str(int(r['packing_list_item_no']))
            for r in rows
            if r.get('packing_list_item_no') is not None and pd.notna(r['packing_list_item_no'])
        })))
        
        row_nos = ', '.join(map(str, sorted(list({
            int(r['source_row_no'])
            for r in rows
            if r.get('source_row_no') is not None
        }))))
        
        pkl_summaries[prod_code] = {
            'product_code': prod_code,
            'total_qty': total_qty,
            'total_net_w': total_net_w,
            'pl_item_nos': item_nos,
            'pl_source_row_nos': row_nos,
        }

    # 2. Get the union of product codes from both PL and INV summaries
    all_product_codes = sorted(list(set(pkl_summaries.keys()) | set(invoice_map.keys())))
    
    rows = []
    for idx, prod_code in enumerate(all_product_codes):
        pkl_sum = pkl_summaries.get(prod_code)
        inv_sum = invoice_map.get(prod_code)
        
        if pkl_sum is None:
            status = 'ROW_MISSING_IN_PL'
            note = 'Product Code exists in Invoice but is missing in Packing List.'
        elif inv_sum is None:
            status = 'PRODUCT_MISSING_IN_INVOICE'
            note = 'Product Code exists in Packing List but is missing in Invoice.'
        else:
            if pkl_sum['total_qty'] == inv_sum['total_qty_shipped']:
                status = 'PASS'
                note = ''
            else:
                status = 'QTY_MISMATCH'
                note = f"Total quantity differs: PL={pkl_sum['total_qty']} vs INV={inv_sum['total_qty_shipped']}."

        inv_item_nos = ''
        inv_source_files = ''
        inv_source_row_nos = ''
        inv_total_qty = None
        
        if inv_sum:
            inv_total_qty = inv_sum['total_qty_shipped']
            sources = inv_sum.get('sources', {})
            inv_item_nos = format_grouped_sources(sources, 'item_nos')
            inv_source_row_nos = format_grouped_sources(sources, 'source_row_nos')
            inv_source_files = ', '.join(sorted(list(sources.keys())))

        rows.append({
            'row_no': idx + 1,
            'product_code': prod_code,
            'pkl_item_nos': pkl_sum['pl_item_nos'] if pkl_sum else None,
            'pkl_source_row_nos': pkl_sum['pl_source_row_nos'] if pkl_sum else None,
            'pkl_total_qty': pkl_sum['total_qty'] if pkl_sum else None,
            'pkl_total_net_weight': pkl_sum['total_net_w'] if pkl_sum else None,
            'inv_item_nos': inv_item_nos if inv_sum else None,
            'inv_source_row_nos': inv_source_row_nos if inv_sum else None,
            'inv_total_qty': inv_total_qty if inv_sum else None,
            'inv_source_files': inv_source_files if inv_sum else None,
            'status': status,
            'note': note,
        })

    return pd.DataFrame(rows)


def build_pl_invoice_not_run_df(note: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'row_no': None,
                'product_code': None,
                'pkl_item_nos': None,
                'pkl_source_row_nos': None,
                'pkl_total_qty': None,
                'pkl_total_net_weight': None,
                'inv_item_nos': None,
                'inv_source_row_nos': None,
                'inv_total_qty': None,
                'inv_source_files': None,
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
                    'co_source_file': None,
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
                    'co_source_file': ', '.join(
                        str(row.get('source_file', ''))
                        for row in matched_co_rows
                    ),
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
                    'co_source_file': co_row.get('source_file'),
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
                'co_source_file': co_row.get('source_file'),
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
                'co_source_file': None,
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
