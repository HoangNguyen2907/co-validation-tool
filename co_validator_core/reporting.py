import pandas as pd


def add_ai_summary_row(
    summary_df: pd.DataFrame,
    ai_result: dict,
    co_df: pd.DataFrame,
    packing_summary_df: pd.DataFrame,
    packing_vs_co_df: pd.DataFrame,
    package_check_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_df = summary_df.copy()
    final_result_df = summary_df[summary_df['metric'] == 'Final result']
    other_df = summary_df[summary_df['metric'] != 'Final result']

    ai_status = ai_result.get('status', 'AI_NOT_RUN')

    if ai_status in ['AI_SUCCESS', 'AI_RATE_LIMITED', 'AI_ERROR']:
        review_status_values = []

        for df in [packing_vs_co_df]:
            if 'ai_review_status' in df.columns:
                review_status_values.extend(
                    status
                    for status in df['ai_review_status'].dropna().tolist()
                    if str(status).strip()
                )

        pl_section = ai_result.get('pl_invoice', {})
        pl_results_df = pl_section.get('results_df', pd.DataFrame())

        if not pl_results_df.empty and 'review_status' in pl_results_df.columns:
            review_status_values.extend(
                status
                for status in pl_results_df['review_status'].dropna().tolist()
                if str(status).strip()
            )

        packing_co_section = ai_result.get('packing_co', {})
        packing_co_results_df = packing_co_section.get(
            'results_df',
            pd.DataFrame(),
        )

        if (
            not packing_co_results_df.empty
            and 'review_status' in packing_co_results_df.columns
        ):
            review_status_values.extend(
                status
                for status in packing_co_results_df[
                    'review_status'
                ].dropna().tolist()
                if str(status).strip()
            )

        package_note = ''

        if 'ai_audit_note' in package_check_df.columns:
            package_note = ' '.join(
                str(note)
                for note in package_check_df['ai_audit_note'].dropna().tolist()
            )

        if ai_status == 'AI_RATE_LIMITED':
            value = 'RATE_LIMITED'
        elif ai_status == 'AI_ERROR':
            value = 'ERROR'
        elif ai_result.get('target_count', 0) == 0:
            value = 'PASS'
        elif pl_results_df.empty and packing_co_results_df.empty:
            value = 'NO_RESULTS'
        elif any(
            status == 'AI_CONFIRMS_RULE_ISSUE'
            for status in review_status_values
        ) or package_note:
            value = 'DIFF_FOUND'
        elif any(
            status in [
                'AI_DISAGREES_WITH_RULE',
                'AI_INSUFFICIENT_CONTEXT',
                'AI_PARSE_UNCLEAR',
                'AI_ERROR',
            ]
            for status in review_status_values
        ):
            value = 'REVIEW'
        elif ai_result.get('skipped_count', 0):
            value = 'LIMITED'
        else:
            value = 'PASS'
    else:
        value = ai_status.replace('AI_', '')

    ai_row = pd.DataFrame(
        [
            {
                'metric': 'AI audit status',
                'value': value,
                'note': (
                    f"Targets: {ai_result.get('target_count', 0)}, "
                    f"audited: {ai_result.get('audited_count', 0)}, "
                    f"skipped: {ai_result.get('skipped_count', 0)}. "
                    f"{ai_result.get('note', '')}"
                ).strip(),
            }
        ]
    )

    return pd.concat([other_df, ai_row, final_result_df], ignore_index=True)


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
