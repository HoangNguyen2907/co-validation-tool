import json

import pandas as pd

from .ai_client import (
    AIResponseParseError,
    extract_json_object,
    get_openrouter_free_model_ids,
    is_openrouter_provider,
    is_rate_limit_error,
    run_ai_chat_completion,
)
from .utils import to_int, to_number


def limit_ai_targets(targets: list[dict], max_targets: int) -> tuple[list[dict], int]:
    ordered_targets = sorted(
        targets,
        key=lambda target: (
            target.get('priority', 999),
            target.get('target_id', ''),
        ),
    )

    if max_targets <= 0:
        return [], len(ordered_targets)

    return ordered_targets[:max_targets], max(0, len(ordered_targets) - max_targets)


def chunk_targets(targets: list[dict], batch_size: int) -> list[list[dict]]:
    batch_size = max(1, int(batch_size))

    return [
        targets[index:index + batch_size]
        for index in range(0, len(targets), batch_size)
    ]


def compact_ai_target(target: dict) -> dict:
    allowed_keys = [
        'target_id',
        'target_type',
        'source_status',
        'product_code',
        'item_no',
        'source_row_no',
        'source_row_nos',
        'packing_total_qty',
        'packing_total_net_weight',
        'rule_quantity_pieces',
        'rule_net_weight_kgs',
        'rule_hs_code',
        'rule_values',
        'pl_row',
        'invoice_row',
        'pkl_context',
        'co_context',
        'text',
        'row',
        'rows',
    ]

    return {
        key: target.get(key)
        for key in allowed_keys
        if key in target and target.get(key) not in [None, '', []]
    }

def extract_ai_pl_invoice_batch(
    targets: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[dict, str]:
    payload = [compact_ai_target(target) for target in targets]
    prompt = f"""
Review only the provided problematic PL vs Invoice validation targets.

Return valid JSON only:
{{
  "reviews": [
    {{
      "target_id": "same target_id from input",
      "review_status": "AI_CONFIRMS_RULE_ISSUE",
      "pl_read": {{
        "product_code": "ABC123",
        "quantity_pieces": 100,
        "evidence": "short exact source text/value"
      }},
      "invoice_read": {{
        "product_code": "ABC123",
        "quantity_pieces": 100,
        "evidence": "short exact source text/value"
      }},
      "assessment": {{
        "issue_type": "QTY_MISMATCH",
        "explanation": "short explanation",
        "confidence": "high"
      }}
    }}
  ],
  "notes": "short batch note"
}}

Rules:
- rule_values are parser outputs for comparison only.
- Do not copy rule_values as source-read values.
- Read PL values only from pl_row raw_rows/parsed.
- Read Invoice values only from invoice_row raw_rows/parsed.
- If source context does not contain a value, return null for that value.
- review_status must be one of AI_CONFIRMS_RULE_ISSUE,
  AI_DISAGREES_WITH_RULE, AI_INSUFFICIENT_CONTEXT, AI_PARSE_UNCLEAR.
- Do not change validation status. Return JSON only.

Targets:
{json.dumps(payload, ensure_ascii=False)}
"""

    raw_response = run_ai_chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You review PL vs Invoice validation issues. '
                    'Return JSON only.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=3000,
    )

    try:
        parsed = extract_json_object(raw_response)
    except Exception as exc:
        raise AIResponseParseError(str(exc), raw_response) from exc

    if not isinstance(parsed, dict):
        raise AIResponseParseError(
            'AI response is not a JSON object.',
            raw_response,
        )

    return parsed, raw_response


def extract_ai_packing_co_batch(
    targets: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[dict, str]:
    payload = [compact_ai_target(target) for target in targets]
    prompt = f"""
Review only the provided problematic Packing vs CO validation targets.

Return valid JSON only:
{{
  "reviews": [
    {{
      "target_id": "same target_id from input",
      "review_status": "AI_CONFIRMS_RULE_ISSUE",
      "pkl_read": {{
        "product_code": "ABC123",
        "quantity_pieces": 100,
        "net_weight_kgs": 25.5,
        "evidence": "short exact source text/value"
      }},
      "co_read": {{
        "item_no": 12,
        "product_code": "ABC123",
        "quantity_pieces": 100,
        "net_weight_kgs": 25.5,
        "hs_code": "123456",
        "evidence": "short exact source text/value"
      }},
      "assessment": {{
        "issue_type": "QTY_MISMATCH",
        "explanation": "short explanation",
        "confidence": "high"
      }}
    }}
  ],
  "notes": "short batch note"
}}

Rules:
- rule_values are parser outputs for comparison only.
- Do not copy rule_values as source-read values.
- Read PKL values only from pkl_context rows/parsed_summary.
- Read CO values from co_context raw_block, matched_items raw_block, or
  search_snippet_by_product_code.
- co_context parsed_by_rule and matched_by_rule are parser trace only.
  Use them to explain parser agreement/disagreement, not as source truth.
- If raw_block does not contain the product code but
  search_snippet_by_product_code points to another item, report
  AI_PARSE_UNCLEAR or AI_DISAGREES_WITH_RULE.
- If source context does not contain a value, return null for that value.
- review_status must be one of AI_CONFIRMS_RULE_ISSUE,
  AI_DISAGREES_WITH_RULE, AI_INSUFFICIENT_CONTEXT, AI_PARSE_UNCLEAR.
- Do not change validation status. Return JSON only.

Targets:
{json.dumps(payload, ensure_ascii=False)}
"""

    raw_response = run_ai_chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You review Packing List vs CO validation issues. '
                    'Return JSON only.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=3000,
    )

    try:
        parsed = extract_json_object(raw_response)
    except Exception as exc:
        raise AIResponseParseError(str(exc), raw_response) from exc

    if not isinstance(parsed, dict):
        raise AIResponseParseError(
            'AI response is not a JSON object.',
            raw_response,
        )

    return parsed, raw_response


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in ['true', 'yes', '1', 'found']


def normalize_review_status(value) -> str:
    status = str(value or '').strip()

    if status in [
        'AI_CONFIRMS_RULE_ISSUE',
        'AI_DISAGREES_WITH_RULE',
        'AI_INSUFFICIENT_CONTEXT',
        'AI_PARSE_UNCLEAR',
        'AI_ERROR',
        'AI_RATE_LIMITED',
    ]:
        return status

    return 'AI_PARSE_UNCLEAR'


def normalize_ai_pl_invoice_reviews(batch_result: dict) -> list[dict]:
    reviews = batch_result.get('reviews', [])

    if not isinstance(reviews, list):
        reviews = []

    rows = []

    for review in reviews:
        if not isinstance(review, dict):
            continue

        pl_read = review.get('pl_read') or {}
        invoice_read = review.get('invoice_read') or {}
        assessment = review.get('assessment') or {}

        rows.append(
            {
                'target_id': str(review.get('target_id') or '').strip(),
                'review_status': normalize_review_status(
                    review.get('review_status')
                ),
                'pl_product_code': str(
                    pl_read.get('product_code') or ''
                ).upper().strip(),
                'pl_quantity_pieces': to_int(pl_read.get('quantity_pieces')),
                'pl_evidence': str(pl_read.get('evidence') or '').strip(),
                'invoice_product_code': str(
                    invoice_read.get('product_code') or ''
                ).upper().strip(),
                'invoice_quantity_pieces': to_int(
                    invoice_read.get('quantity_pieces')
                ),
                'invoice_evidence': str(
                    invoice_read.get('evidence') or ''
                ).strip(),
                'issue_type': str(assessment.get('issue_type') or '').strip(),
                'explanation': str(assessment.get('explanation') or '').strip(),
                'confidence': str(assessment.get('confidence') or '').strip(),
                'ai_error': '',
            }
        )

    return rows


def normalize_ai_packing_co_reviews(batch_result: dict) -> list[dict]:
    reviews = batch_result.get('reviews', [])

    if not isinstance(reviews, list):
        reviews = []

    rows = []

    for review in reviews:
        if not isinstance(review, dict):
            continue

        pkl_read = review.get('pkl_read') or {}
        co_read = review.get('co_read') or {}
        assessment = review.get('assessment') or {}

        rows.append(
            {
                'target_id': str(review.get('target_id') or '').strip(),
                'review_status': normalize_review_status(
                    review.get('review_status')
                ),
                'pkl_product_code': str(
                    pkl_read.get('product_code') or ''
                ).upper().strip(),
                'pkl_quantity_pieces': to_int(
                    pkl_read.get('quantity_pieces')
                ),
                'pkl_net_weight_kgs': to_number(
                    pkl_read.get('net_weight_kgs')
                ),
                'pkl_evidence': str(pkl_read.get('evidence') or '').strip(),
                'co_item_no': to_int(co_read.get('item_no')),
                'co_product_code': str(
                    co_read.get('product_code') or ''
                ).upper().strip(),
                'co_quantity_pieces': to_int(
                    co_read.get('quantity_pieces')
                ),
                'co_net_weight_kgs': to_number(
                    co_read.get('net_weight_kgs')
                ),
                'co_hs_code': str(co_read.get('hs_code') or '').strip(),
                'co_evidence': str(co_read.get('evidence') or '').strip(),
                'issue_type': str(assessment.get('issue_type') or '').strip(),
                'explanation': str(assessment.get('explanation') or '').strip(),
                'confidence': str(assessment.get('confidence') or '').strip(),
                'ai_error': '',
            }
        )

    return rows


def normalize_ai_target_results(batch_result: dict) -> list[dict]:
    items = batch_result.get('targets', [])

    if not isinstance(items, list):
        items = []

    rows = []

    for item in items:
        if not isinstance(item, dict):
            continue
        target_type = str(item.get('target_type') or '').strip()
        legacy_quantity = item.get('quantity_pieces')
        legacy_weight = item.get('net_weight_kgs')
        legacy_is_packing = target_type in [
            'packing_product',
            'packing_excluded_row',
            'packing_packages',
        ]

        rows.append(
            {
                'target_id': str(item.get('target_id') or '').strip(),
                'target_type': target_type,
                'found': to_bool(item.get('found')),
                'product_code': str(item.get('product_code') or '').upper().strip(),
                'item_no': to_int(item.get('item_no')),
                'hs_code': str(item.get('hs_code') or '').strip(),
                'co_quantity_pieces': to_int(
                    item.get(
                        'co_quantity_pieces',
                        None if legacy_is_packing else legacy_quantity,
                    )
                ),
                'co_net_weight_kgs': to_number(
                    item.get(
                        'co_net_weight_kgs',
                        None if legacy_is_packing else legacy_weight,
                    )
                ),
                'pkl_quantity_pieces': to_int(
                    item.get(
                        'pkl_quantity_pieces',
                        legacy_quantity if legacy_is_packing else None,
                    )
                ),
                'pkl_net_weight_kgs': to_number(
                    item.get(
                        'pkl_net_weight_kgs',
                        legacy_weight if legacy_is_packing else None,
                    )
                ),
                'total_packages': to_int(item.get('total_packages')),
                'evidence': str(item.get('evidence') or '').strip(),
                'note': str(item.get('note') or '').strip(),
                'ai_audit_status': 'AI_SUCCESS',
            }
        )

    return rows

def get_empty_pl_invoice_review_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'target_id',
            'review_status',
            'pl_product_code',
            'pl_quantity_pieces',
            'pl_evidence',
            'invoice_product_code',
            'invoice_quantity_pieces',
            'invoice_evidence',
            'issue_type',
            'explanation',
            'confidence',
            'ai_error',
        ]
    )


def get_empty_packing_co_review_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            'target_id',
            'review_status',
            'pkl_product_code',
            'pkl_quantity_pieces',
            'pkl_net_weight_kgs',
            'pkl_evidence',
            'co_item_no',
            'co_product_code',
            'co_quantity_pieces',
            'co_net_weight_kgs',
            'co_hs_code',
            'co_evidence',
            'issue_type',
            'explanation',
            'confidence',
            'ai_error',
        ]
    )


def build_error_review_row(target: dict, columns: list[str], error_text: str) -> dict:
    row = {column: '' for column in columns}
    row['target_id'] = target.get('target_id', '')
    row['review_status'] = (
        'AI_RATE_LIMITED' if is_rate_limit_error(error_text) else 'AI_ERROR'
    )
    row['ai_error'] = error_text

    return row


def run_ai_review_section(
    section_name: str,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    targets: list[dict],
    batch_size: int,
    max_targets: int,
    empty_df: pd.DataFrame,
    extract_batch_func,
    normalize_func,
) -> dict:
    limited_targets, skipped_count = limit_ai_targets(targets, max_targets)

    if not limited_targets:
        return {
            'status': 'AI_SUCCESS',
            'note': f'No problematic rows selected for {section_name}.',
            'targets': [],
            'target_count': 0,
            'audited_count': 0,
            'skipped_count': skipped_count,
            'results_df': empty_df.copy(),
            'errors': [],
            'raw_responses': [],
            'failed_batches': [],
            'model_used': model,
            'fallback_models': [],
        }

    result_rows = []
    errors = []
    raw_responses = []
    failed_batches = []
    status = 'AI_SUCCESS'
    active_model = model
    fallback_models = []
    fallback_note = ''

    for batch_index, batch in enumerate(chunk_targets(limited_targets, batch_size), start=1):
        target_ids = [target['target_id'] for target in batch]

        try:
            batch_result, raw_response = extract_batch_func(
                targets=batch,
                api_key=api_key,
                base_url=base_url,
                model=active_model,
            )
            raw_responses.append(
                {
                    'batch_index': batch_index,
                    'target_ids': target_ids,
                    'model': active_model,
                    'raw_response': raw_response,
                }
            )
            result_rows.extend(normalize_func(batch_result))
        except Exception as exc:
            error_text = str(exc)
            raw_error_response = getattr(exc, 'raw_response', '')

            if (
                is_openrouter_provider(provider, base_url)
                and is_rate_limit_error(error_text)
            ):
                if not fallback_models:
                    try:
                        fallback_models = get_openrouter_free_model_ids(
                            api_key=api_key,
                            exclude_model=active_model,
                            limit=8,
                        )
                    except Exception as fallback_list_exc:
                        fallback_models = []
                        errors.append(str(fallback_list_exc))

                fallback_success = False

                for fallback_model in fallback_models:
                    try:
                        batch_result, raw_response = extract_batch_func(
                            targets=batch,
                            api_key=api_key,
                            base_url=base_url,
                            model=fallback_model,
                        )
                        active_model = fallback_model
                        fallback_success = True
                        fallback_note = (
                            f'Fallback model used after OpenRouter 429: '
                            f'{fallback_model}.'
                        )
                        raw_responses.append(
                            {
                                'batch_index': batch_index,
                                'target_ids': target_ids,
                                'model': active_model,
                                'raw_response': raw_response,
                            }
                        )
                        result_rows.extend(normalize_func(batch_result))
                        break
                    except Exception as fallback_exc:
                        if not is_rate_limit_error(str(fallback_exc)):
                            continue

                if fallback_success:
                    continue

            failed_batches.append(
                {
                    'batch_index': batch_index,
                    'target_ids': target_ids,
                    'model': active_model,
                    'fallback_models': fallback_models,
                    'error': error_text,
                    'raw_response': raw_error_response,
                }
            )
            errors.append(f'{section_name} AI batch {batch_index} failed: {error_text}')

            if is_rate_limit_error(error_text):
                status = 'AI_RATE_LIMITED'
            else:
                status = 'AI_ERROR'

            for target in batch:
                result_rows.append(
                    build_error_review_row(
                        target,
                        list(empty_df.columns),
                        error_text,
                    )
                )

            if is_rate_limit_error(error_text):
                break

    results_df = pd.DataFrame(result_rows, columns=empty_df.columns)
    note_parts = []

    if skipped_count:
        note_parts.append(f'{skipped_count} target(s) skipped by max target limit.')

    if fallback_note:
        note_parts.append(fallback_note)

    if errors:
        note_parts.extend(errors[:3])

    return {
        'status': status,
        'note': ' '.join(note_parts),
        'targets': limited_targets,
        'target_count': len(limited_targets),
        'audited_count': len(results_df),
        'skipped_count': skipped_count,
        'results_df': results_df,
        'errors': errors,
        'raw_responses': raw_responses,
        'failed_batches': failed_batches,
        'model_used': active_model,
        'fallback_models': fallback_models,
    }


def build_ai_review_result(
    enabled: bool,
    connection_ready: bool,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    pl_invoice_targets: list[dict],
    packing_co_targets: list[dict],
    batch_size: int,
    max_targets: int,
) -> dict:
    empty_pl_df = get_empty_pl_invoice_review_df()
    empty_packing_co_df = get_empty_packing_co_review_df()
    empty_section = {
        'status': 'AI_NOT_RUN',
        'note': 'AI review is disabled.',
        'targets': [],
        'target_count': 0,
        'audited_count': 0,
        'skipped_count': 0,
        'errors': [],
        'raw_responses': [],
        'failed_batches': [],
        'model_used': model,
        'fallback_models': [],
    }

    if not enabled:
        pl_inv_section = {**empty_section, 'results_df': empty_pl_df}
        packing_co_section = {**empty_section, 'results_df': empty_packing_co_df}
        return {
            'status': 'AI_NOT_RUN',
            'note': 'AI review is disabled.',
            'target_count': 0,
            'audited_count': 0,
            'skipped_count': 0,
            'model_used': model,
            'fallback_models': [],
            'pl_invoice': pl_inv_section,
            'packing_co': packing_co_section,
        }

    if not connection_ready:
        pl_inv_section = {
            **empty_section,
            'status': 'AI_ERROR',
            'note': 'AI connection has not been tested successfully.',
            'targets': pl_invoice_targets,
            'target_count': len(pl_invoice_targets),
            'results_df': empty_pl_df,
            'errors': ['AI connection has not been tested successfully.'],
        }
        packing_co_section = {
            **empty_section,
            'status': 'AI_ERROR',
            'note': 'AI connection has not been tested successfully.',
            'targets': packing_co_targets,
            'target_count': len(packing_co_targets),
            'results_df': empty_packing_co_df,
            'errors': ['AI connection has not been tested successfully.'],
        }
        return {
            'status': 'AI_ERROR',
            'note': 'AI connection has not been tested successfully.',
            'target_count': len(pl_invoice_targets) + len(packing_co_targets),
            'audited_count': 0,
            'skipped_count': 0,
            'model_used': model,
            'fallback_models': [],
            'pl_invoice': pl_inv_section,
            'packing_co': packing_co_section,
        }

    pl_inv_section = run_ai_review_section(
        section_name='PL vs Invoice',
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        targets=pl_invoice_targets,
        batch_size=batch_size,
        max_targets=max_targets,
        empty_df=empty_pl_df,
        extract_batch_func=extract_ai_pl_invoice_batch,
        normalize_func=normalize_ai_pl_invoice_reviews,
    )
    packing_co_section = run_ai_review_section(
        section_name='Packing vs CO',
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=pl_inv_section.get('model_used') or model,
        targets=packing_co_targets,
        batch_size=batch_size,
        max_targets=max_targets,
        empty_df=empty_packing_co_df,
        extract_batch_func=extract_ai_packing_co_batch,
        normalize_func=normalize_ai_packing_co_reviews,
    )
    sections = [pl_inv_section, packing_co_section]
    statuses = [section.get('status') for section in sections]

    if 'AI_RATE_LIMITED' in statuses:
        status = 'AI_RATE_LIMITED'
    elif 'AI_ERROR' in statuses:
        status = 'AI_ERROR'
    else:
        status = 'AI_SUCCESS'

    notes = [
        str(section.get('note') or '').strip()
        for section in sections
        if str(section.get('note') or '').strip()
    ]
    fallback_models = []

    for section in sections:
        for fallback_model in section.get('fallback_models') or []:
            if fallback_model not in fallback_models:
                fallback_models.append(fallback_model)

    return {
        'status': status,
        'note': ' '.join(notes),
        'target_count': sum(section.get('target_count', 0) for section in sections),
        'audited_count': sum(section.get('audited_count', 0) for section in sections),
        'skipped_count': sum(section.get('skipped_count', 0) for section in sections),
        'model_used': packing_co_section.get('model_used') or pl_inv_section.get('model_used') or model,
        'fallback_models': fallback_models,
        'pl_invoice': pl_inv_section,
        'packing_co': packing_co_section,
    }
