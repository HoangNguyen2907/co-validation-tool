from numbers import Number

import pandas as pd

from .utils import to_int, to_number
from .validation import tokenize_description


def format_ai_value(value) -> str:
    if value is None or pd.isna(value):
        return 'blank'

    return str(value)


def append_diff_note(
    notes: list[str],
    label: str,
    rule_value,
    ai_value,
    tolerance: float = 0,
):
    if rule_value is None or ai_value is None:
        return

    if isinstance(rule_value, Number) and isinstance(ai_value, Number):
        if abs(float(rule_value) - float(ai_value)) <= tolerance:
            return
    elif str(rule_value).strip() == str(ai_value).strip():
        return

    notes.append(
        f'{label}: rule {format_ai_value(rule_value)} vs AI {format_ai_value(ai_value)}'
    )


def find_ai_matches_for_co_row(row, ai_items_df: pd.DataFrame) -> pd.DataFrame:
    if ai_items_df.empty:
        return ai_items_df

    item_no = to_int(row.get('item_no'))

    if item_no is None:
        return ai_items_df.iloc[0:0]

    return ai_items_df[ai_items_df['item_no'] == item_no]


def find_ai_matches_for_product(
    product_code: str,
    co_item_no,
    ai_items_df: pd.DataFrame,
) -> pd.DataFrame:
    if ai_items_df.empty:
        return ai_items_df

    item_no = to_int(co_item_no)

    if item_no is not None:
        item_matches = ai_items_df[ai_items_df['item_no'] == item_no]

        if not item_matches.empty:
            return item_matches

    product_code = str(product_code).upper().strip()

    return ai_items_df[
        ai_items_df.apply(
            lambda row: (
                product_code in row.get('product_codes', [])
                or product_code in tokenize_description(row.get('description', ''))
            ),
            axis=1,
        )
    ]


def get_ai_result_by_id(ai_results_df: pd.DataFrame, target_id: str):
    if ai_results_df.empty or 'target_id' not in ai_results_df.columns:
        return None

    matches = ai_results_df[ai_results_df['target_id'] == target_id]

    if matches.empty:
        return None

    return matches.iloc[0]


def get_review_result_by_id(results_df: pd.DataFrame, target_id: str):
    if results_df.empty or 'target_id' not in results_df.columns:
        return None

    matches = results_df[results_df['target_id'] == target_id]

    if matches.empty:
        return None

    return matches.iloc[0]


def build_pl_invoice_review_note(result) -> str:
    if result is None:
        return ''

    status = str(result.get('review_status') or '').strip()
    error = str(result.get('ai_error') or '').strip()

    if status in ['AI_ERROR', 'AI_RATE_LIMITED']:
        return error

    parts = []
    pl_qty = result.get('pl_quantity_pieces')
    invoice_qty = result.get('invoice_quantity_pieces')
    pl_product = str(result.get('pl_product_code') or '').strip()
    invoice_product = str(result.get('invoice_product_code') or '').strip()
    explanation = str(result.get('explanation') or '').strip()

    if pl_product or invoice_product:
        parts.append(
            f'AI product read: PL {pl_product or "blank"} '
            f'vs Invoice {invoice_product or "blank"}'
        )

    if pl_qty is not None or invoice_qty is not None:
        parts.append(
            f'AI qty read: PL {format_ai_value(pl_qty)} '
            f'vs Invoice {format_ai_value(invoice_qty)}'
        )

    if explanation:
        parts.append(explanation)

    return '; '.join(parts)


def build_packing_co_review_note(result) -> str:
    if result is None:
        return ''

    status = str(result.get('review_status') or '').strip()
    error = str(result.get('ai_error') or '').strip()

    if status in ['AI_ERROR', 'AI_RATE_LIMITED']:
        return error

    parts = []
    pkl_qty = result.get('pkl_quantity_pieces')
    co_qty = result.get('co_quantity_pieces')
    pkl_weight = result.get('pkl_net_weight_kgs')
    co_weight = result.get('co_net_weight_kgs')
    co_item_no = result.get('co_item_no')
    explanation = str(result.get('explanation') or '').strip()

    if pkl_qty is not None or co_qty is not None:
        parts.append(
            f'AI qty read: PKL {format_ai_value(pkl_qty)} '
            f'vs CO {format_ai_value(co_qty)}'
        )

    if pkl_weight is not None or co_weight is not None:
        parts.append(
            f'AI weight read: PKL {format_ai_value(pkl_weight)} '
            f'vs CO {format_ai_value(co_weight)}'
        )

    if co_item_no is not None:
        parts.append(f'AI CO item {format_ai_value(co_item_no)}')

    if explanation:
        parts.append(explanation)

    return '; '.join(parts)


def add_ai_review_columns(
    pl_vs_invoice_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
    ai_result: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pl_vs_invoice_df = pl_vs_invoice_df.copy()
    packing_vs_co_df = packing_vs_co_df.copy()
    pl_results_df = (
        ai_result.get('pl_invoice', {}).get('results_df', pd.DataFrame())
    )
    packing_co_results_df = (
        ai_result.get('packing_co', {}).get('results_df', pd.DataFrame())
    )
    pl_statuses = []
    pl_notes = []

    for _, row in pl_vs_invoice_df.iterrows():
        row_no = to_int(row.get('row_no'))
        result = (
            None
            if row_no is None
            else get_review_result_by_id(
                pl_results_df,
                f'pl_invoice_row:{row_no}',
            )
        )

        if result is None:
            pl_statuses.append('')
            pl_notes.append('')
            continue

        pl_statuses.append(result.get('review_status') or '')
        pl_notes.append(build_pl_invoice_review_note(result))

    packing_statuses = []
    packing_notes = []

    for _, row in packing_vs_co_df.iterrows():
        product_code = str(row.get('product_code') or '').upper().strip()
        result = (
            None
            if not product_code
            else get_review_result_by_id(
                packing_co_results_df,
                f'packing_co_product:{product_code}',
            )
        )

        if result is None:
            packing_statuses.append('')
            packing_notes.append('')
            continue

        packing_statuses.append(result.get('review_status') or '')
        packing_notes.append(build_packing_co_review_note(result))

    pl_vs_invoice_df['ai_review_status'] = pl_statuses
    pl_vs_invoice_df['ai_review_note'] = pl_notes
    packing_vs_co_df['ai_review_status'] = packing_statuses
    packing_vs_co_df['ai_review_note'] = packing_notes

    return pl_vs_invoice_df, packing_vs_co_df


def build_ai_note_from_result(result) -> str:
    if result is None:
        return ''

    note = str(result.get('note') or '').strip()
    evidence = str(result.get('evidence') or '').strip()
    status = str(result.get('ai_audit_status') or '').strip()

    if status in ['AI_ERROR', 'AI_RATE_LIMITED']:
        return note

    if evidence:
        return f'{note} Evidence: {evidence}'.strip()

    return note


def join_ai_notes(parts: list[str], result=None) -> str:
    base_note = build_ai_note_from_result(result) if result is not None else ''
    all_parts = [part for part in parts if str(part).strip()]

    if base_note:
        all_parts.append(base_note)

    return '; '.join(all_parts)


def add_ai_audit_columns(
    co_df: pd.DataFrame,
    packing_summary_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
    package_check_df: pd.DataFrame,
    ai_result: dict,
    packing_packages,
    weight_tolerance: float,
):
    co_df = co_df.copy()
    packing_summary_df = packing_summary_df.copy()
    packing_vs_co_df = packing_vs_co_df.copy()
    package_check_df = package_check_df.copy()

    ai_results_df = ai_result.get('results_df', pd.DataFrame())
    ai_status = ai_result.get('status', 'AI_NOT_RUN')
    ai_note = ai_result.get('note', '')

    co_statuses = []
    co_notes = []

    for _, row in co_df.iterrows():
        item_no = to_int(row.get('item_no'))
        target_id = f'co_item:{item_no}' if item_no is not None else ''
        result = get_ai_result_by_id(ai_results_df, target_id)

        if result is None:
            co_statuses.append('')
            co_notes.append('')
            continue

        result_status = result.get('ai_audit_status') or ai_status

        if result_status != 'AI_SUCCESS':
            co_statuses.append(result_status)
            co_notes.append(build_ai_note_from_result(result) or ai_note)
            continue

        if not result.get('found'):
            co_statuses.append('AI_NOT_FOUND')
            co_notes.append(build_ai_note_from_result(result))
            continue

        notes = []
        append_diff_note(
            notes,
            'CO qty',
            row.get('quantity_pieces'),
            result.get('co_quantity_pieces'),
        )
        append_diff_note(
            notes,
            'CO weight',
            row.get('net_weight_kgs'),
            result.get('co_net_weight_kgs'),
            tolerance=weight_tolerance,
        )
        append_diff_note(
            notes,
            'HS',
            row.get('hs_code'),
            result.get('hs_code'),
        )

        if notes:
            co_statuses.append('AI_DIFF')
            co_notes.append(join_ai_notes(notes, result))
        else:
            co_statuses.append('AI_MATCH')
            co_notes.append(build_ai_note_from_result(result))

    co_df['ai_audit_status'] = co_statuses
    co_df['ai_audit_note'] = co_notes

    summary_statuses = []
    summary_notes = []

    for _, row in packing_summary_df.iterrows():
        product_code = str(row.get('product_code') or '').upper().strip()
        result = get_ai_result_by_id(
            ai_results_df,
            f'packing_product:{product_code}',
        )

        if result is None:
            summary_statuses.append('')
            summary_notes.append('')
            continue

        result_status = result.get('ai_audit_status') or ai_status

        if result_status != 'AI_SUCCESS':
            summary_statuses.append(result_status)
            summary_notes.append(build_ai_note_from_result(result) or ai_note)
            continue

        notes = []
        append_diff_note(
            notes,
            'PKL qty',
            row.get('total_shipped_qty'),
            result.get('pkl_quantity_pieces'),
        )
        append_diff_note(
            notes,
            'PKL weight',
            row.get('total_net_weight'),
            result.get('pkl_net_weight_kgs'),
            tolerance=weight_tolerance,
        )

        if notes:
            summary_statuses.append('AI_DIFF')
            summary_notes.append(join_ai_notes(notes, result))
        else:
            summary_statuses.append('AI_MATCH')
            summary_notes.append(build_ai_note_from_result(result))

    packing_summary_df['ai_audit_status'] = summary_statuses
    packing_summary_df['ai_audit_note'] = summary_notes

    packing_statuses = []
    packing_notes = []

    for _, row in packing_vs_co_df.iterrows():
        product_code = str(row.get('product_code') or '').upper().strip()
        co_item_no = to_int(row.get('co_item_no'))
        candidate_ids = []

        if co_item_no is not None:
            candidate_ids.append(f'co_item:{co_item_no}')

        candidate_ids.append(f'co_missing_product:{product_code}')
        candidate_ids.append(f'packing_product:{product_code}')

        result = None
        co_result = None
        pkl_result = None

        for target_id in candidate_ids:
            result = get_ai_result_by_id(ai_results_df, target_id)

            if result is not None:
                break

        if co_item_no is not None:
            co_result = get_ai_result_by_id(ai_results_df, f'co_item:{co_item_no}')

        if co_result is None:
            co_result = get_ai_result_by_id(
                ai_results_df,
                f'co_missing_product:{product_code}',
            )

        pkl_result = get_ai_result_by_id(
            ai_results_df,
            f'packing_product:{product_code}',
        )

        if result is None:
            packing_statuses.append('')
            packing_notes.append('')
            continue

        result_status = result.get('ai_audit_status') or ai_status

        if result_status != 'AI_SUCCESS':
            packing_statuses.append(result_status)
            packing_notes.append(build_ai_note_from_result(result) or ai_note)
            continue

        notes = []
        co_qty = None if co_result is None else co_result.get('co_quantity_pieces')
        co_weight = None if co_result is None else co_result.get('co_net_weight_kgs')
        pkl_qty = (
            None if pkl_result is None
            else pkl_result.get('pkl_quantity_pieces')
        )
        pkl_weight = (
            None if pkl_result is None
            else pkl_result.get('pkl_net_weight_kgs')
        )

        if co_qty is not None:
            notes.append(f'AI CO qty {format_ai_value(co_qty)}')

        if pkl_qty is not None:
            notes.append(f'AI PKL qty {format_ai_value(pkl_qty)}')

        if co_qty is not None and pkl_qty is not None and co_qty != pkl_qty:
            notes.append(
                f'AI qty mismatch: CO {format_ai_value(co_qty)} '
                f'vs PKL {format_ai_value(pkl_qty)}'
            )
        else:
            append_diff_note(
                notes,
                'PKL qty',
                row.get('packing_total_qty'),
                pkl_qty,
            )
            append_diff_note(
                notes,
                'CO qty',
                row.get('co_quantity_pieces'),
                co_qty,
            )

        if co_weight is not None:
            notes.append(f'AI CO weight {format_ai_value(co_weight)}')

        if pkl_weight is not None:
            notes.append(f'AI PKL weight {format_ai_value(pkl_weight)}')

        if (
            co_weight is not None
            and pkl_weight is not None
            and abs(float(co_weight) - float(pkl_weight)) > weight_tolerance
        ):
            notes.append(
                f'AI weight mismatch: CO {format_ai_value(co_weight)} '
                f'vs PKL {format_ai_value(pkl_weight)}'
            )
        else:
            append_diff_note(
                notes,
                'PKL weight',
                row.get('packing_total_net_weight'),
                pkl_weight,
                tolerance=weight_tolerance,
            )
            append_diff_note(
                notes,
                'CO weight',
                row.get('co_net_weight_kgs'),
                co_weight,
                tolerance=weight_tolerance,
            )

        if any('mismatch' in note for note in notes):
            packing_statuses.append('AI_DIFF')
            packing_notes.append(join_ai_notes(notes, result))
        elif notes:
            packing_statuses.append('AI_MATCH')
            packing_notes.append(join_ai_notes(notes, result))
        elif result.get('found'):
            packing_statuses.append('AI_MATCH')
            packing_notes.append(build_ai_note_from_result(result))
        else:
            packing_statuses.append('AI_NOT_FOUND')
            packing_notes.append(build_ai_note_from_result(result))

    packing_vs_co_df['ai_audit_status'] = packing_statuses
    packing_vs_co_df['ai_audit_note'] = packing_notes

    co_package_result = get_ai_result_by_id(ai_results_df, 'co_package_total')
    packing_package_result = get_ai_result_by_id(ai_results_df, 'packing_packages')
    ai_co_total_packages = (
        None if co_package_result is None
        else to_int(co_package_result.get('total_packages'))
    )
    ai_packing_packages = (
        None if packing_package_result is None
        else to_int(packing_package_result.get('total_packages'))
    )

    package_check_df['ai_co_total_packages'] = ai_co_total_packages
    package_check_df['ai_packing_packages'] = ai_packing_packages

    package_notes = []

    if co_package_result is not None:
        if co_package_result.get('ai_audit_status') != 'AI_SUCCESS':
            package_notes.append(build_ai_note_from_result(co_package_result))
        elif ai_co_total_packages is None:
            package_notes.append('AI did not return CO total packages.')
        elif (
            'co_total_packages' in package_check_df.columns
            and package_check_df['co_total_packages'].notna().any()
            and int(package_check_df['co_total_packages'].dropna().iloc[0])
            != ai_co_total_packages
        ):
            package_notes.append('AI CO total packages differs from rule extraction.')

    if packing_package_result is not None:
        if packing_package_result.get('ai_audit_status') != 'AI_SUCCESS':
            package_notes.append(build_ai_note_from_result(packing_package_result))
        elif ai_packing_packages is None:
            package_notes.append('AI did not return Packing List packages.')
        elif packing_packages is not None and packing_packages != ai_packing_packages:
            package_notes.append('AI Packing List packages differs from rule extraction.')

    package_check_df['ai_audit_note'] = ' '.join(
        note for note in package_notes if note
    )

    return co_df, packing_summary_df, packing_vs_co_df, package_check_df
