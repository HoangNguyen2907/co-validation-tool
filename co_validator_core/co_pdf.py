import io
import re

import fitz
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image

from .constants import HS_RE, ITEM_START_RE, QTY_RE, WEIGHT_RE
from .utils import normalize_text, parse_number_text


def repair_item_no(raw_item_no, previous_item_no, next_raw_item_no=None):
    expected = previous_item_no + 1 if previous_item_no is not None else None

    if raw_item_no.isdigit():
        current = int(raw_item_no)

        if expected is None:
            return current, False

        if current == expected:
            return current, False

        if str(expected).endswith(str(current)):
            return expected, True

        if next_raw_item_no and next_raw_item_no.isdigit():
            if int(next_raw_item_no) == expected + 1:
                return expected, True

        return current, False

    if expected is not None:
        if next_raw_item_no and next_raw_item_no.isdigit():
            if int(next_raw_item_no) == expected + 1:
                return expected, True

        return expected, True

    return None, False


def iter_item_blocks(text: str):
    matches = list(ITEM_START_RE.finditer(text))
    previous_item_no = None

    for index, match in enumerate(matches):
        raw_item_no = match.group(1)

        next_raw_item_no = (
            matches[index + 1].group(1)
            if index + 1 < len(matches)
            else None
        )

        item_no, repaired = repair_item_no(
            raw_item_no,
            previous_item_no,
            next_raw_item_no,
        )

        if item_no is None:
            continue

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = normalize_text(text[start:end])

        yield item_no, block, raw_item_no, repaired

        previous_item_no = item_no


def extract_text_pdfplumber(pdf_path: str) -> str:
    parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or '')

    return normalize_text('\n'.join(parts))


def extract_text_ocr(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []

    for page in doc:
        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes('png')))

        text = pytesseract.image_to_string(
            img,
            lang='eng',
            config='--psm 6',
        )

        parts.append(text)

    return normalize_text('\n'.join(parts))


def should_use_ocr(text: str) -> bool:
    if len(text.strip()) < 500:
        return True

    if 'HS Code' not in text and 'HS CODE' not in text:
        return True

    return False


def get_pdf_text(pdf_path: str) -> str:
    text = extract_text_pdfplumber(pdf_path)

    if should_use_ocr(text):
        text = extract_text_ocr(pdf_path)

    return text


def extract_weight(block: str, decimal_separator: str = 'auto'):
    match = WEIGHT_RE.search(block)

    if match:
        return parse_number_text(match.group(1), decimal_separator=decimal_separator)

    weight_pos = re.search(r'WEIGHT', block, re.I)

    if weight_pos:
        tail = block[weight_pos.end():]
        match = re.search(r'([\d,.]+)\s*KGS?\b', tail, re.I)

        if match:
            return parse_number_text(match.group(1), decimal_separator=decimal_separator)

    return None


def clean_description(block: str) -> str:
    desc = HS_RE.split(block)[0]

    desc = re.sub(
        r'\b\d{2}[A-Z]{2,}\d+\b.*$',
        '',
        desc,
        flags=re.I,
    )

    desc = re.sub(
        r'\b(WO|PE|CTH|PSR|\d+%)\b\s+\d+(?:,\d+)?\s*PIECES.*$',
        '',
        desc,
        flags=re.I,
    )

    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip()


def extract_co_items(text: str, decimal_separator: str = 'auto') -> pd.DataFrame:
    rows = []

    text = re.sub(
        r'1\. Parties which accept this form.*?(?=Attachment|Original|Page \d+ of \d+|$)',
        '',
        text,
        flags=re.S,
    )

    for item_no, block, raw_item_no, repaired in iter_item_blocks(text):
        hs = HS_RE.search(block)
        qty = QTY_RE.search(block)

        rows.append(
            {
                'item_no': item_no,
                'raw_item_no': raw_item_no,
                'item_no_repaired': repaired,
                'raw_block': block,
                'description': clean_description(block),
                'hs_code': hs.group(1) if hs else '',
                'quantity_pieces': (
                    int(qty.group(1).replace(',', ''))
                    if qty
                    else None
                ),
                'net_weight_kgs': extract_weight(
                    block,
                    decimal_separator=decimal_separator,
                ),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df = (
            df
            .drop_duplicates(subset=['item_no'])
            .sort_values('item_no')
            .reset_index(drop=True)
        )

    return df


def extract_co_total_packages(text: str):
    matches = re.findall(
        r'\bTOTAL\b[\s\S]{0,300}?\((\d+)\)',
        text,
        flags=re.I,
    )

    if not matches:
        return None

    return int(matches[-1])
