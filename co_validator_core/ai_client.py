import json
import ssl
from urllib import request
from urllib.error import HTTPError, URLError

from .constants import AI_PROVIDER_CONFIGS


class AIResponseParseError(Exception):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response

def get_ai_base_url(provider: str, custom_base_url: str) -> str:
    if provider == 'Custom OpenAI-compatible':
        return custom_base_url.strip()

    return AI_PROVIDER_CONFIGS[provider]['base_url']


def get_ai_default_model(provider: str) -> str:
    return AI_PROVIDER_CONFIGS[provider]['default_model']


def is_openrouter_provider(provider: str, base_url: str) -> bool:
    return (
        provider == 'OpenRouter'
        or base_url.rstrip('/') == AI_PROVIDER_CONFIGS['OpenRouter']['base_url']
    )


def is_rate_limit_error(error_text: str) -> bool:
    return '429' in error_text or 'rate limit' in error_text.lower()


def pricing_value_is_zero(value) -> bool:
    if value is None:
        return False

    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return str(value).strip() == '0'


def score_openrouter_validator_model(model: dict) -> tuple[int, list[str]]:
    model_id = str(model.get('id') or '').lower()
    model_name = str(model.get('name') or '').lower()
    supported_parameters = set(model.get('supported_parameters') or [])
    context_length = int(model.get('context_length') or 0)
    score = 0
    reasons = []
    reliable_families = [
        ('qwen/', 'Qwen family'),
        ('deepseek/', 'DeepSeek family'),
        ('google/', 'Google/Gemini family'),
        ('mistralai/', 'Mistral family'),
        ('meta-llama/', 'Llama family'),
        ('moonshotai/', 'Moonshot/Kimi family'),
        ('openrouter/', 'OpenRouter first-party route'),
    ]

    for index, (pattern, label) in enumerate(reliable_families):
        if pattern in model_id or pattern.replace('/', '') in model_name:
            score += 100 - index * 5
            reasons.append(f'preferred reliable model family: {label}')
            break

    if 'structured_outputs' in supported_parameters:
        score += 20
        reasons.append('supports structured_outputs')

    if 'response_format' in supported_parameters:
        score += 15
        reasons.append('supports response_format')

    if context_length >= 32768:
        score += 10
        reasons.append('enough context for validator batches')

    if model_id.endswith(':free'):
        score += 5
        reasons.append('explicit free model')

    return score, reasons


def list_openrouter_free_models(api_key: str = '') -> list[dict]:
    headers = {'Content-Type': 'application/json'}

    if api_key.strip():
        headers['Authorization'] = f'Bearer {api_key.strip()}'

    req = request.Request(
        'https://openrouter.ai/api/v1/models',
        headers=headers,
    )

    try:
        try:
            import certifi

            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_context = ssl.create_default_context()

        with request.urlopen(req, timeout=20, context=ssl_context) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Cannot fetch OpenRouter models: {exc}') from exc

    models = payload.get('data', [])

    if not isinstance(models, list):
        return []

    free_models = []

    for model in models:
        if not isinstance(model, dict):
            continue

        pricing = model.get('pricing') or {}
        architecture = model.get('architecture') or {}
        output_modalities = architecture.get('output_modalities') or []

        if (
            pricing_value_is_zero(pricing.get('prompt'))
            and pricing_value_is_zero(pricing.get('completion'))
            and 'text' in output_modalities
        ):
            score, reasons = score_openrouter_validator_model(model)
            model['_validator_score'] = score
            model['_validator_score_reasons'] = reasons
            free_models.append(model)

    return sorted(
        free_models,
        key=lambda model: (
            -int(model.get('_validator_score') or 0),
            -int(model.get('context_length') or 0),
            str(model.get('id', '')),
        ),
    )


def get_openrouter_free_model_ids(
    api_key: str = '',
    exclude_model: str = '',
    limit: int = 8,
) -> list[str]:
    free_models = list_openrouter_free_models(api_key=api_key)
    exclude_model = exclude_model.strip()
    model_ids = []

    for model in free_models:
        model_id = str(model.get('id') or '').strip()

        if not model_id or model_id == exclude_model:
            continue

        model_ids.append(model_id)

        if len(model_ids) >= limit:
            break

    return model_ids


def describe_openrouter_free_model_candidates(
    api_key: str = '',
    limit: int = 8,
) -> list[dict]:
    models = list_openrouter_free_models(api_key=api_key)
    rows = []

    for model in models[:limit]:
        rows.append(
            {
                'id': model.get('id'),
                'name': model.get('name'),
                'context_length': model.get('context_length'),
                'validator_score': model.get('_validator_score'),
                'reasons': ', '.join(model.get('_validator_score_reasons') or []),
            }
        )

    return rows


def run_ai_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 1200,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            'The openai package is not installed. Run pip install -r requirements.txt.'
        ) from exc

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30,
        max_retries=0,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content or ''


def test_ai_connection(api_key: str, base_url: str, model: str) -> str:
    return run_ai_chat_completion(
        api_key=api_key,
        base_url=base_url,
        model=model,
        messages=[
            {
                'role': 'user',
                'content': 'Return exactly: OK',
            },
        ],
        max_tokens=10,
    )


def test_ai_connection_with_fallback(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    try:
        test_ai_connection(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        return {
            'ok': True,
            'model': model,
            'note': 'AI connection succeeded.',
            'fallback_models': [],
        }
    except Exception as exc:
        error_text = str(exc)

        if not (
            is_openrouter_provider(provider, base_url)
            and is_rate_limit_error(error_text)
        ):
            raise

        fallback_models = get_openrouter_free_model_ids(
            api_key=api_key,
            exclude_model=model,
            limit=8,
        )

        for fallback_model in fallback_models:
            try:
                test_ai_connection(
                    api_key=api_key,
                    base_url=base_url,
                    model=fallback_model,
                )
                return {
                    'ok': True,
                    'model': fallback_model,
                    'note': (
                        f'AI connection succeeded with fallback model '
                        f'{fallback_model}. Original model was rate limited.'
                    ),
                    'fallback_models': fallback_models,
                }
            except Exception as fallback_exc:
                if not is_rate_limit_error(str(fallback_exc)):
                    continue

        raise RuntimeError(
            'OpenRouter selected model was rate limited and no free fallback '
            'model passed the connection test.'
        ) from exc


def extract_json_object(text: str) -> dict:
    text = text.strip()

    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')

        if start == -1 or end == -1 or end <= start:
            raise

        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text_without_trailing_commas = re.sub(r',\s*([}\]])', r'\1', text)

        return json.loads(text_without_trailing_commas)
