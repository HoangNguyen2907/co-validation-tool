import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_secrets import decrypt_api_key
from ai_secrets import encrypt_api_key
from ai_secrets import get_ai_key_encryption_secret
from ai_secrets import get_ai_local_storage_key
from ai_secrets import get_browser_local_storage_item
from ai_secrets import get_provider_secret_key
from ai_secrets import remove_browser_local_storage_item
from ai_secrets import set_browser_local_storage_item
from co_validator_core.ai_client import (
    describe_openrouter_free_model_candidates,
    get_ai_base_url,
    get_ai_default_model,
    is_openrouter_provider,
    test_ai_connection_with_fallback,
)
from co_validator_core.ai_review import (
    add_ai_review_columns,
    build_ai_review_result,
    build_packing_co_review_targets,
    build_pl_invoice_review_targets,
)
from co_validator_core.co_pdf import (
    extract_co_items,
    extract_co_total_packages,
    get_pdf_text,
)
from co_validator_core.constants import AI_PROVIDER_CONFIGS, AI_REVIEW_SCOPES
from co_validator_core.excel_inputs import prepare_invoice_df, prepare_packing_df
from co_validator_core.reporting import (
    add_ai_summary_row,
    build_empty_co_df,
    build_summary,
    write_report,
)
from co_validator_core.validation import (
    build_package_check_df,
    build_package_check_not_run_df,
    build_packing_summary,
    build_packing_vs_co_not_run_df,
    build_pl_invoice_not_run_df,
    compare_pl_invoice,
    validate_packing_vs_co,
)


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

st.subheader('AI Review Audit')

use_ai_audit = st.checkbox(
    'Use AI to audit problematic rows',
    value=False,
)

ai_document_scope = AI_REVIEW_SCOPES[0]
ai_max_targets = 50
ai_batch_size = 10
ai_provider = list(AI_PROVIDER_CONFIGS.keys())[0]
ai_custom_base_url = ''
ai_base_url = get_ai_base_url(ai_provider, ai_custom_base_url)
ai_model = get_ai_default_model(ai_provider)
ai_key_encryption_secret = ''
ai_storage_key = ''
saved_encrypted_ai_key = ''
saved_ai_api_key = ''
provider_secret_api_key = ''
ai_api_key = ''
effective_ai_api_key = ''

if use_ai_audit:
    ai_document_scope = st.selectbox(
        'AI review scope',
        options=AI_REVIEW_SCOPES,
    )

    ai_max_targets = st.number_input(
        'AI max targets',
        min_value=1,
        max_value=200,
        value=50,
        step=1,
    )

    ai_batch_size = st.number_input(
        'AI batch size',
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    ai_provider = st.selectbox(
        'AI provider',
        options=list(AI_PROVIDER_CONFIGS.keys()),
    )

    if ai_provider == 'Custom OpenAI-compatible':
        ai_custom_base_url = st.text_input(
            'AI base URL',
            value='',
            placeholder='https://example.com/v1',
        )

    ai_base_url = get_ai_base_url(ai_provider, ai_custom_base_url)

    ai_model = st.text_input(
        'AI model',
        value=get_ai_default_model(ai_provider),
    )

    ai_key_encryption_secret = get_ai_key_encryption_secret()
    ai_storage_key = get_ai_local_storage_key(ai_provider, ai_base_url)

    if ai_key_encryption_secret:
        saved_encrypted_ai_key = (
            get_browser_local_storage_item(
                ai_storage_key,
                key=f'ai_saved_key_read_{ai_storage_key}',
            )
            or ''
        )

    saved_ai_api_key = decrypt_api_key(
        saved_encrypted_ai_key,
        ai_key_encryption_secret,
    )
    provider_secret_api_key = get_provider_secret_key(ai_provider)

    ai_api_key = st.text_input(
        'AI API token',
        value='',
        type='password',
        placeholder='Leave blank to use a saved browser key or app secret',
    )
    effective_ai_api_key = (
        ai_api_key.strip()
        or saved_ai_api_key
        or provider_secret_api_key
    )

    if ai_api_key.strip():
        st.caption('AI key source: manual input for this session.')
    elif saved_ai_api_key:
        st.caption('AI key source: encrypted key saved in this browser.')
    elif provider_secret_api_key:
        st.caption('AI key source: Streamlit Cloud secrets/env.')
    else:
        st.caption('AI key source: not configured.')

    save_disabled = (
        not ai_api_key.strip()
        or not ai_key_encryption_secret
    )
    forget_disabled = not bool(saved_encrypted_ai_key)
    save_col, forget_col = st.columns(2)

    with save_col:
        if st.button(
            'Save API key on this browser',
            disabled=save_disabled,
        ):
            encrypted_api_key = encrypt_api_key(
                ai_api_key.strip(),
                ai_key_encryption_secret,
            )
            set_browser_local_storage_item(
                ai_storage_key,
                encrypted_api_key,
                key=f'ai_saved_key_write_{ai_storage_key}',
            )
            st.success('Encrypted API key saved in this browser.')

    with forget_col:
        if st.button(
            'Forget saved API key',
            disabled=forget_disabled,
        ):
            remove_browser_local_storage_item(
                ai_storage_key,
                key=f'ai_saved_key_remove_{ai_storage_key}',
            )
            st.success('Saved API key removed from this browser.')

    if not ai_key_encryption_secret:
        st.caption(
            'Remember key is disabled. Configure AI_KEY_ENCRYPTION_SECRET '
            'in Streamlit secrets/env to enable encrypted browser storage.'
        )

    ai_connection_key = (
        f'{ai_provider}|{ai_base_url}|{ai_model}|{len(effective_ai_api_key or "")}'
    )

    if st.session_state.get('ai_connection_key') != ai_connection_key:
        st.session_state.ai_connection_ready = False
        st.session_state.ai_connection_note = ''
        st.session_state.ai_runtime_model = ''
        st.session_state.ai_fallback_models = []

    ai_test_disabled = (
        not effective_ai_api_key.strip()
        or not ai_model.strip()
        or not ai_base_url.strip()
    )

    if st.button(
        'Test AI connection',
        disabled=ai_test_disabled,
    ):
        with st.spinner('Testing AI connection...'):
            try:
                connection_result = test_ai_connection_with_fallback(
                    provider=ai_provider,
                    api_key=effective_ai_api_key,
                    base_url=ai_base_url,
                    model=ai_model,
                )
                st.session_state.ai_connection_ready = True
                st.session_state.ai_connection_key = ai_connection_key
                st.session_state.ai_connection_note = connection_result['note']
                st.session_state.ai_runtime_model = connection_result['model']
                st.session_state.ai_fallback_models = connection_result[
                    'fallback_models'
                ]
            except Exception as exc:
                st.session_state.ai_connection_ready = False
                st.session_state.ai_connection_key = ai_connection_key
                st.session_state.ai_connection_note = f'AI connection failed: {exc}'
                st.session_state.ai_runtime_model = ''
                st.session_state.ai_fallback_models = []

    if st.session_state.get('ai_connection_ready'):
        st.success(st.session_state.get('ai_connection_note'))
        runtime_model_note = st.session_state.get('ai_runtime_model')

        if runtime_model_note and runtime_model_note != ai_model:
            st.info(f'AI audit will use fallback model: {runtime_model_note}')
    elif st.session_state.get('ai_connection_note'):
        st.error(st.session_state.get('ai_connection_note'))
    else:
        st.info('Test the AI connection before validating to enable AI audit.')

    if is_openrouter_provider(ai_provider, ai_base_url):
        if st.button(
            'List OpenRouter free fallback models',
            disabled=not effective_ai_api_key.strip(),
        ):
            try:
                st.session_state.openrouter_free_candidates = (
                    describe_openrouter_free_model_candidates(
                        api_key=effective_ai_api_key,
                        limit=8,
                    )
                )
            except Exception as exc:
                st.session_state.openrouter_free_candidates = []
                st.error(f'Cannot list OpenRouter free models: {exc}')

        candidates = st.session_state.get('openrouter_free_candidates', [])

        if candidates:
            st.dataframe(
                pd.DataFrame(candidates),
                hide_index=True,
                width='stretch',
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

        pl_df, packing_packages, packing_context = prepare_packing_df(
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
            invoice_context = {'raw_df': pd.DataFrame()}
            pl_vs_invoice_df = build_pl_invoice_not_run_df(
                'No Invoice available. PL vs Invoice validation was not run.'
            )
        else:
            inv_df, invoice_context = prepare_invoice_df(
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

        pl_invoice_targets = []
        packing_co_targets = []

        if use_ai_audit:
            if ai_document_scope in [
                'PL vs Invoice + Packing vs CO',
                'PL vs Invoice only',
            ]:
                pl_invoice_targets = build_pl_invoice_review_targets(
                    pl_vs_invoice_df,
                    packing_context,
                    invoice_context,
                )

            if (
                co_pdf_file is not None
                and ai_document_scope in [
                    'PL vs Invoice + Packing vs CO',
                    'Packing vs CO only',
                ]
            ):
                packing_co_targets = build_packing_co_review_targets(
                    co_text,
                    co_df,
                    packing_context,
                    packing_summary_df,
                    packing_vs_co_df,
                )

        ai_result = build_ai_review_result(
            enabled=use_ai_audit,
            connection_ready=st.session_state.get('ai_connection_ready', False),
            provider=ai_provider,
            api_key=effective_ai_api_key,
            base_url=ai_base_url,
            model=st.session_state.get('ai_runtime_model') or ai_model,
            pl_invoice_targets=pl_invoice_targets,
            packing_co_targets=packing_co_targets,
            batch_size=int(ai_batch_size),
            max_targets=int(ai_max_targets),
        )

        (
            pl_vs_invoice_df,
            packing_vs_co_df,
        ) = add_ai_review_columns(
            pl_vs_invoice_df,
            packing_vs_co_df,
            ai_result=ai_result,
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
        summary_df = add_ai_summary_row(
            summary_df,
            ai_result,
            co_df,
            packing_summary_df,
            packing_vs_co_df,
            package_check_df,
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

    if use_ai_audit:
        with st.expander('AI review details'):
            st.write(
                {
                    'status': ai_result.get('status'),
                    'note': ai_result.get('note'),
                    'target_count': ai_result.get('target_count'),
                    'audited_count': ai_result.get('audited_count'),
                    'skipped_count': ai_result.get('skipped_count'),
                    'base_url': ai_base_url,
                    'model': ai_model,
                    'model_used': ai_result.get('model_used'),
                    'fallback_models': ai_result.get('fallback_models'),
                }
            )

            for section_key, section_title in [
                ('pl_invoice', 'PL vs Invoice AI review'),
                ('packing_co', 'Packing vs CO AI review'),
            ]:
                section = ai_result.get(section_key, {})
                st.subheader(section_title)
                st.write(
                    {
                        'status': section.get('status'),
                        'note': section.get('note'),
                        'target_count': section.get('target_count'),
                        'audited_count': section.get('audited_count'),
                        'skipped_count': section.get('skipped_count'),
                        'model_used': section.get('model_used'),
                    }
                )

                targets = section.get('targets') or []

                if targets:
                    st.write('Targets sent to AI')
                    st.json(targets)

                results_df = section.get('results_df', pd.DataFrame())

                if not results_df.empty:
                    st.write('AI review result')
                    st.dataframe(results_df, hide_index=True, width='stretch')
                elif not targets:
                    st.info(f'No {section_title} details available.')

                failed_batches = section.get('failed_batches', [])

                if failed_batches:
                    st.write('AI failed batches')

                    for failed_batch in failed_batches:
                        st.write(
                            {
                                'batch_index': failed_batch.get('batch_index'),
                                'target_ids': failed_batch.get('target_ids'),
                                'model': failed_batch.get('model'),
                                'fallback_models': failed_batch.get(
                                    'fallback_models'
                                ),
                                'error': failed_batch.get('error'),
                            }
                        )

                        raw_response = failed_batch.get('raw_response') or ''

                        if raw_response:
                            st.text_area(
                                'Raw AI response',
                                value=raw_response,
                                height=300,
                                key=(
                                    f"ai_failed_raw_{section_key}_"
                                    f"{failed_batch.get('batch_index')}"
                                ),
                            )

                raw_responses = section.get('raw_responses', [])

                if raw_responses:
                    st.write('Successful raw AI responses')

                    for raw_response_info in raw_responses:
                        st.write(
                            {
                                'batch_index': raw_response_info.get(
                                    'batch_index'
                                ),
                                'target_ids': raw_response_info.get('target_ids'),
                                'model': raw_response_info.get('model'),
                            }
                        )
                        st.text_area(
                            'Raw AI response',
                            value=raw_response_info.get('raw_response') or '',
                            height=220,
                            key=(
                                f"ai_success_raw_{section_key}_"
                                f"{raw_response_info.get('batch_index')}"
                            ),
                        )

    # st.download_button(
    #     label='Download validation report',
    #     data=output_path.read_bytes(),
    #     file_name='co_validation_result.xlsx',
    #     mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    # )

    # with st.expander('Preview CO extracted text'):
    #     st.text(co_text)
