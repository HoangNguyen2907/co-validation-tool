import base64
import hashlib
import json
import os

import streamlit as st


def get_secret_value(key: str) -> str:
    try:
        value = st.secrets.get(key, '')
    except Exception:
        value = ''

    return str(value or os.environ.get(key, '') or '').strip()


def get_provider_secret_key(provider: str) -> str:
    provider_secret_names = {
        'OpenRouter': 'OPENROUTER_API_KEY',
        'Moonshot/Kimi': 'MOONSHOT_API_KEY',
        'Gemini OpenAI-compatible': 'GEMINI_API_KEY',
    }
    secret_name = provider_secret_names.get(provider, 'AI_API_KEY')

    return get_secret_value(secret_name) or get_secret_value('AI_API_KEY')


def get_ai_key_encryption_secret() -> str:
    return get_secret_value('AI_KEY_ENCRYPTION_SECRET')


def get_ai_local_storage_key(provider: str, base_url: str) -> str:
    storage_hash = hashlib.sha256(
        f'{provider}|{base_url}'.encode('utf-8')
    ).hexdigest()[:16]

    return f'co_validation_ai_key_v1_{storage_hash}'


def get_fernet_from_secret(secret: str):
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError(
            'The cryptography package is not installed. '
            'Run pip install -r requirements.txt.'
        ) from exc

    key = base64.urlsafe_b64encode(
        hashlib.sha256(secret.encode('utf-8')).digest()
    )

    return Fernet(key)


def encrypt_api_key(api_key: str, secret: str) -> str:
    fernet = get_fernet_from_secret(secret)
    payload = {
        'version': 1,
        'api_key': api_key,
    }

    return fernet.encrypt(
        json.dumps(payload).encode('utf-8')
    ).decode('utf-8')


def decrypt_api_key(encrypted_value: str, secret: str) -> str:
    if not encrypted_value or not secret:
        return ''

    try:
        fernet = get_fernet_from_secret(secret)
        raw_payload = fernet.decrypt(encrypted_value.encode('utf-8'))
        payload = json.loads(raw_payload.decode('utf-8'))
    except Exception:
        return ''

    if int(payload.get('version') or 0) != 1:
        return ''

    return str(payload.get('api_key') or '').strip()


def browser_local_storage_eval(js_expression: str, key: str):
    try:
        from streamlit_js_eval import streamlit_js_eval
    except ImportError:
        return None

    return streamlit_js_eval(
        js_expressions=js_expression,
        key=key,
    )


def get_browser_local_storage_item(storage_key: str, key: str):
    return browser_local_storage_eval(
        f'localStorage.getItem({json.dumps(storage_key)})',
        key=key,
    )


def set_browser_local_storage_item(storage_key: str, value: str, key: str):
    return browser_local_storage_eval(
        (
            f'localStorage.setItem({json.dumps(storage_key)}, '
            f'{json.dumps(value)}); true'
        ),
        key=key,
    )


def remove_browser_local_storage_item(storage_key: str, key: str):
    return browser_local_storage_eval(
        f'localStorage.removeItem({json.dumps(storage_key)}); true',
        key=key,
    )
