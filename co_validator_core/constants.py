import re

ITEM_START_RE = re.compile(
    r'(?m)(?:^|\n)\s*(\S{1,3})\s*(?:[|/]\s*)?N/M\s+'
)

HS_RE = re.compile(r'HS\s*Code\s*:\s*(\d+)', re.I)

QTY_RE = re.compile(r'(\d+(?:,\d+)?)\s*PIECES', re.I)

WEIGHT_RE = re.compile(
    r'N\s*\.?\s*W\s*E\s*I\s*G\s*H\s*T\s*[:\s\n]*([\d,.]+)\s*KGS?\b',
    re.I,
)

AI_PROVIDER_CONFIGS = {
    'OpenRouter': {
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'moonshotai/kimi-k2.6:free',
    },
    'Moonshot/Kimi': {
        'base_url': 'https://api.moonshot.ai/v1',
        'default_model': 'kimi-k2.6',
    },
    'Gemini OpenAI-compatible': {
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai',
        'default_model': 'gemini-2.5-flash',
    },
    'Custom OpenAI-compatible': {
        'base_url': '',
        'default_model': '',
    },
}

AI_REVIEW_SCOPES = [
    'PL vs Invoice + Packing vs CO',
    'PL vs Invoice only',
    'Packing vs CO only',
]

AI_PROBLEM_STATUSES = {
    'CO_VALUE_MISSING',
    'NOT_FOUND_IN_CO',
    'DUPLICATE_IN_CO',
    'QTY_MISMATCH',
    'WEIGHT_MISMATCH',
    'QTY_AND_WEIGHT_MISMATCH',
}

AI_TARGET_PRIORITIES = {
    'pl_invoice_mismatch': 1,
    'packing_co_mismatch': 2,
    'co_package_total': 1,
    'packing_packages': 2,
    'co_item_missing_value': 3,
    'packing_excluded_row': 4,
    'co_missing_product': 5,
    'co_duplicate': 6,
    'packing_product_mismatch': 7,
}

INTERNAL_SOURCE_ROW_NO = '__source_row_no'
