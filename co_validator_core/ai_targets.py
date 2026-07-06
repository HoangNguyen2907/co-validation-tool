import re

import pandas as pd

from .constants import AI_PROBLEM_STATUSES, AI_TARGET_PRIORITIES, INTERNAL_SOURCE_ROW_NO
from .utils import normalize_text, to_int


def row_to_context(row) -> dict:
    return {
        str(key): '' if pd.isna(value) else str(value)
        for key, value in row.items()
        if str(key) != INTERNAL_SOURCE_ROW_NO
    }


def parse_source_row_nos(value) -> list[int]:
    row_nos = []

    for item in str(value or '').split(','):
        row_no = to_int(item.strip())

        if row_no is not None:
            row_nos.append(row_no)

    return row_nos


def get_raw_rows_by_source_row_nos(
    raw_df: pd.DataFrame,
    source_row_nos,
) -> list[dict]:
    lookup = build_raw_rows_by_source_row_no(raw_df)
    return get_raw_rows_from_lookup(lookup, source_row_nos)


def build_raw_rows_by_source_row_no(raw_df: pd.DataFrame) -> dict[int, list[dict]]:
    if raw_df.empty or INTERNAL_SOURCE_ROW_NO not in raw_df.columns:
        return {}

    lookup: dict[int, list[dict]] = {}

    for _, raw_row in raw_df.iterrows():
        row_no = to_int(raw_row.get(INTERNAL_SOURCE_ROW_NO))

        if row_no is None:
            continue

        lookup.setdefault(row_no, []).append(
            {
                'source_row_no': row_no,
                'raw_row': row_to_context(raw_row),
            }
        )

    return lookup


def get_raw_rows_from_lookup(
    raw_rows_by_source_row_no: dict[int, list[dict]],
    source_row_nos,
) -> list[dict]:
    rows = []

    for source_row_no in source_row_nos:
        row_no = to_int(source_row_no)

        if row_no is None:
            continue

        rows.extend(raw_rows_by_source_row_no.get(row_no, []))

    return rows


def find_text_snippet(text: str, pattern: str, window: int = 1200) -> str:
    if not text.strip() or not str(pattern).strip():
        return ''

    match = re.search(re.escape(str(pattern).strip()), text, flags=re.I)

    if not match:
        return ''

    start = max(0, match.start() - window // 2)
    end = min(len(text), match.end() + window // 2)

    return normalize_text(text[start:end])


def find_package_snippet(text: str, window: int = 2500) -> str:
    if not text.strip():
        return ''

    matches = list(
        re.finditer(r'\b(TOTAL|PACKAGES?|PACKAGE)\b', text, flags=re.I)
    )

    if not matches:
        return normalize_text(text[-window:])

    match = matches[-1]
    start = max(0, match.start() - window // 2)
    end = min(len(text), match.end() + window // 2)

    return normalize_text(text[start:end])


def append_ai_target(targets: list[dict], target: dict):
    existing_ids = {item['target_id'] for item in targets}

    if target['target_id'] not in existing_ids:
        targets.append(target)


def build_co_audit_targets(
    co_text: str,
    co_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
    package_check_df: pd.DataFrame,
) -> list[dict]:
    targets = []
    co_rows_by_item = {
        to_int(row.get('item_no')): row
        for _, row in co_df.iterrows()
        if to_int(row.get('item_no')) is not None
    }

    for _, row in co_df.iterrows():
        item_no = to_int(row.get('item_no'))
        missing_value = (
            pd.isna(row.get('quantity_pieces'))
            or pd.isna(row.get('net_weight_kgs'))
            or str(row.get('hs_code') or '').strip() == ''
        )

        if item_no is None or not missing_value:
            continue

        append_ai_target(
            targets,
            {
                'target_id': f'co_item:{item_no}',
                'target_type': 'co_item',
                'priority': AI_TARGET_PRIORITIES['co_item_missing_value'],
                'source_section': 'CO',
                'source_status': 'CO_VALUE_MISSING',
                'item_no': item_no,
                'product_code': '',
                'rule_quantity_pieces': row.get('quantity_pieces'),
                'rule_net_weight_kgs': row.get('net_weight_kgs'),
                'rule_hs_code': row.get('hs_code'),
                'text': row.get('raw_block', ''),
            },
        )

    for _, row in packing_vs_co_df.iterrows():
        status = str(row.get('status') or '')

        if status not in AI_PROBLEM_STATUSES:
            continue

        product_code = str(row.get('product_code') or '').upper().strip()
        co_item_no = to_int(row.get('co_item_no'))

        if co_item_no is not None:
            co_rule_row = co_rows_by_item.get(co_item_no)
            target_id = f'co_item:{co_item_no}'
            target_type = 'co_item'
            priority_key = (
                'co_duplicate' if status == 'DUPLICATE_IN_CO'
                else 'co_item_missing_value'
            )
            text = (
                str(co_rule_row.get('raw_block') or '')
                if co_rule_row is not None
                else str(row.get('co_description') or '')
            )
            rule_quantity_pieces = (
                None
                if co_rule_row is None
                else co_rule_row.get('quantity_pieces')
            )
            rule_net_weight_kgs = (
                None
                if co_rule_row is None
                else co_rule_row.get('net_weight_kgs')
            )
            rule_hs_code = (
                ''
                if co_rule_row is None
                else co_rule_row.get('hs_code')
            )
        else:
            target_id = f'co_missing_product:{product_code}'
            target_type = 'co_missing_product'
            priority_key = 'co_missing_product'
            text = find_text_snippet(co_text, product_code)
            rule_quantity_pieces = None
            rule_net_weight_kgs = None
            rule_hs_code = ''

        append_ai_target(
            targets,
            {
                'target_id': target_id,
                'target_type': target_type,
                'priority': AI_TARGET_PRIORITIES[priority_key],
                'source_section': 'Packing_vs_CO',
                'source_status': status,
                'item_no': co_item_no,
                'product_code': product_code,
                'rule_quantity_pieces': rule_quantity_pieces,
                'rule_net_weight_kgs': rule_net_weight_kgs,
                'rule_hs_code': rule_hs_code,
                'text': text,
            },
        )

    if (
        not package_check_df.empty
        and str(package_check_df.iloc[0].get('status') or '')
        in ['CO_TOTAL_PACKAGES_NOT_FOUND', 'PACKAGE_MISMATCH']
    ):
        append_ai_target(
            targets,
            {
                'target_id': 'co_package_total',
                'target_type': 'co_package_total',
                'priority': AI_TARGET_PRIORITIES['co_package_total'],
                'source_section': 'Package_Check',
                'source_status': package_check_df.iloc[0].get('status'),
                'text': find_package_snippet(co_text),
            },
        )

    return targets


def build_packing_audit_targets(
    packing_context: dict,
    packing_summary_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
    package_check_df: pd.DataFrame,
) -> list[dict]:
    targets = []
    raw_df = packing_context.get('raw_df', pd.DataFrame())
    excluded_rows_df = packing_context.get('excluded_rows_df', pd.DataFrame())
    package_context_df = packing_context.get('package_context_df', pd.DataFrame())
    raw_rows_by_source_row_no = build_raw_rows_by_source_row_no(raw_df)
    problem_product_codes = {
        str(row.get('product_code') or '').upper().strip()
        for _, row in packing_vs_co_df.iterrows()
        if str(row.get('status') or '') in AI_PROBLEM_STATUSES
    }

    for _, row in packing_summary_df.iterrows():
        product_code = str(row.get('product_code') or '').upper().strip()

        if product_code not in problem_product_codes:
            continue

        source_row_numbers = [
            to_int(value)
            for value in str(row.get('excel_file_row_nos') or '').split(',')
        ]
        source_row_numbers = [
            value for value in source_row_numbers if value is not None
        ]
        context_rows = get_raw_rows_from_lookup(
            raw_rows_by_source_row_no,
            source_row_numbers,
        )

        append_ai_target(
            targets,
            {
                'target_id': f'packing_product:{product_code}',
                'target_type': 'packing_product',
                'priority': AI_TARGET_PRIORITIES['packing_product_mismatch'],
                'source_section': 'Packing_Summary',
                'source_status': 'PACKING_PRODUCT_REVIEW',
                'product_code': product_code,
                'source_row_nos': source_row_numbers,
                'packing_total_qty': row.get('total_shipped_qty'),
                'packing_total_net_weight': row.get('total_net_weight'),
                'rows': context_rows,
            },
        )

    for _, row in excluded_rows_df.iterrows():
        source_row_no = to_int(row.get(INTERNAL_SOURCE_ROW_NO))

        if source_row_no is None:
            continue

        append_ai_target(
            targets,
            {
                'target_id': f'packing_excluded_row:{source_row_no}',
                'target_type': 'packing_excluded_row',
                'priority': AI_TARGET_PRIORITIES['packing_excluded_row'],
                'source_section': 'Packing_List',
                'source_status': 'PACKING_ROW_EXCLUDED',
                'source_row_no': source_row_no,
                'row': row_to_context(row),
            },
        )

    if (
        not package_check_df.empty
        and str(package_check_df.iloc[0].get('status') or '')
        in ['PACKING_PACKAGES_NOT_FOUND', 'PACKAGE_MISMATCH']
    ):
        append_ai_target(
            targets,
            {
                'target_id': 'packing_packages',
                'target_type': 'packing_packages',
                'priority': AI_TARGET_PRIORITIES['packing_packages'],
                'source_section': 'Package_Check',
                'source_status': package_check_df.iloc[0].get('status'),
                'rows': [
                    row_to_context(row)
                    for _, row in package_context_df.iterrows()
                ],
            },
        )

    return targets


def build_pl_invoice_review_targets(
    pl_vs_invoice_df: pd.DataFrame,
    packing_context: dict,
    invoice_context: dict,
) -> list[dict]:
    targets = []
    packing_raw_df = packing_context.get('raw_df', pd.DataFrame())
    invoice_raw_df = invoice_context.get('raw_df', pd.DataFrame())
    packing_raw_rows_by_source_row_no = build_raw_rows_by_source_row_no(
        packing_raw_df
    )
    invoice_raw_rows_by_source_row_no = build_raw_rows_by_source_row_no(
        invoice_raw_df
    )

    for _, row in pl_vs_invoice_df.iterrows():
        status = str(row.get('status') or '')

        if status in ['PASS', 'NOT_RUN', '']:
            continue

        row_no = to_int(row.get('row_no'))

        if row_no is None:
            continue

        pl_source_row_no = to_int(row.get('pl_source_row_no'))
        inv_source_row_no = to_int(row.get('inv_source_row_no'))
        pl_raw_rows = get_raw_rows_from_lookup(
            packing_raw_rows_by_source_row_no,
            [pl_source_row_no],
        )
        invoice_raw_rows = get_raw_rows_from_lookup(
            invoice_raw_rows_by_source_row_no,
            [inv_source_row_no],
        )

        append_ai_target(
            targets,
            {
                'target_id': f'pl_invoice_row:{row_no}',
                'target_type': 'pl_invoice_row',
                'priority': AI_TARGET_PRIORITIES['pl_invoice_mismatch'],
                'source_section': 'PL_vs_Invoice',
                'source_status': status,
                'rule_values': {
                    'pl_product_code': row.get('pl_product_code'),
                    'invoice_product_code': row.get('inv_product_code'),
                    'pl_quantity_pieces': row.get('pl_qty_shipped'),
                    'invoice_quantity_pieces': row.get('inv_qty_shipped'),
                    'note': row.get('note'),
                },
                'pl_row': (
                    None if pl_source_row_no is None else {
                        'source_row_no': pl_source_row_no,
                        'parsed': {
                            'product_code': row.get('pl_product_code'),
                            'quantity_pieces': row.get('pl_qty_shipped'),
                        },
                        'raw_rows': pl_raw_rows,
                    }
                ),
                'invoice_row': (
                    None if inv_source_row_no is None else {
                        'source_row_no': inv_source_row_no,
                        'parsed': {
                            'product_code': row.get('inv_product_code'),
                            'quantity_pieces': row.get('inv_qty_shipped'),
                        },
                        'raw_rows': invoice_raw_rows,
                    }
                ),
            },
        )

    return targets


def build_packing_co_review_targets(
    co_text: str,
    co_df: pd.DataFrame,
    packing_context: dict,
    packing_summary_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
) -> list[dict]:
    targets = []
    packing_raw_df = packing_context.get('raw_df', pd.DataFrame())
    packing_raw_rows_by_source_row_no = build_raw_rows_by_source_row_no(
        packing_raw_df
    )
    summary_by_product = {
        str(row.get('product_code') or '').upper().strip(): row
        for _, row in packing_summary_df.iterrows()
    }
    co_rows_by_item = {
        to_int(row.get('item_no')): row
        for _, row in co_df.iterrows()
        if to_int(row.get('item_no')) is not None
    }

    for _, row in packing_vs_co_df.iterrows():
        status = str(row.get('status') or '')

        if status in ['PASS', 'NOT_RUN', '']:
            continue

        product_code = str(row.get('product_code') or '').upper().strip()

        if not product_code:
            continue

        summary_row = summary_by_product.get(product_code)
        source_row_nos = (
            []
            if summary_row is None
            else parse_source_row_nos(summary_row.get('excel_file_row_nos'))
        )
        pkl_rows = get_raw_rows_from_lookup(
            packing_raw_rows_by_source_row_no,
            source_row_nos,
        )
        co_item_nos = parse_source_row_nos(row.get('co_item_no'))
        search_snippet_by_product_code = find_text_snippet(
            co_text,
            product_code,
        )
        co_context = {
            'matched_by_rule': {
                'co_item_no': None,
                'method': 'product_code_token_index',
                'confidence': 'rule_trace',
            },
            'found_by_rule': False,
            'search_product_code': product_code,
            'search_snippet_by_product_code': search_snippet_by_product_code,
        }

        if len(co_item_nos) == 1:
            co_item_no = co_item_nos[0]
            co_row = co_rows_by_item.get(co_item_no)
            co_context = {
                'matched_by_rule': {
                    'co_item_no': co_item_no,
                    'method': 'product_code_token_index',
                    'confidence': 'rule_trace',
                },
                'found_by_rule': co_row is not None,
                'raw_block': (
                    '' if co_row is None else str(co_row.get('raw_block') or '')
                ),
                'description': (
                    '' if co_row is None else str(co_row.get('description') or '')
                ),
                'parsed_by_rule': {
                    'quantity_pieces': (
                        None if co_row is None
                        else co_row.get('quantity_pieces')
                    ),
                    'net_weight_kgs': (
                        None if co_row is None
                        else co_row.get('net_weight_kgs')
                    ),
                    'hs_code': '' if co_row is None else co_row.get('hs_code'),
                },
                'search_product_code': product_code,
                'search_snippet_by_product_code': search_snippet_by_product_code,
            }
        elif len(co_item_nos) > 1:
            matched_items = []

            for co_item_no in co_item_nos:
                co_row = co_rows_by_item.get(co_item_no)

                if co_row is None:
                    continue

                matched_items.append(
                    {
                        'co_item_no': co_item_no,
                        'raw_block': str(co_row.get('raw_block') or ''),
                        'description': str(co_row.get('description') or ''),
                        'parsed_by_rule': {
                            'quantity_pieces': co_row.get('quantity_pieces'),
                            'net_weight_kgs': co_row.get('net_weight_kgs'),
                            'hs_code': co_row.get('hs_code'),
                        },
                    }
                )

            co_context = {
                'matched_by_rule': {
                    'co_item_no': co_item_nos,
                    'method': 'product_code_token_index',
                    'confidence': 'rule_trace',
                },
                'found_by_rule': bool(matched_items),
                'matched_items': matched_items,
                'search_product_code': product_code,
                'search_snippet_by_product_code': search_snippet_by_product_code,
            }

        append_ai_target(
            targets,
            {
                'target_id': f'packing_co_product:{product_code}',
                'target_type': 'packing_co_product',
                'priority': AI_TARGET_PRIORITIES['packing_co_mismatch'],
                'source_section': 'Packing_vs_CO',
                'source_status': status,
                'product_code': product_code,
                'rule_values': {
                    'product_code': product_code,
                    'packing_total_qty': row.get('packing_total_qty'),
                    'co_quantity_pieces': row.get('co_quantity_pieces'),
                    'qty_diff': row.get('qty_diff'),
                    'packing_total_net_weight': row.get(
                        'packing_total_net_weight'
                    ),
                    'co_net_weight_kgs': row.get('co_net_weight_kgs'),
                    'weight_diff': row.get('weight_diff'),
                    'note': row.get('note'),
                },
                'pkl_context': {
                    'excel_file_row_nos': source_row_nos,
                    'rows': pkl_rows,
                    'parsed_summary': (
                        {}
                        if summary_row is None
                        else {
                            'total_shipped_qty': summary_row.get(
                                'total_shipped_qty'
                            ),
                            'total_net_weight': summary_row.get(
                                'total_net_weight'
                            ),
                        }
                    ),
                },
                'co_context': co_context,
            },
        )

    return targets
