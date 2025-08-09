import re

import pandas as pd
from transformers import LlamaTokenizer
from datasets import Dataset
import structlog

from project_secrets import hf_token

WINDOW_SIZE = 256
STRIDE = 128

logger = structlog.get_logger()


def extract_page_data(lines):
    """Extracts data about pages from the IVTFF file lines."""
    page_regex = re.compile(r'<f(\d+[rv]\d*|Ros)>')
    page_lines = [l for l in lines if page_regex.match(l)]

    illustration_map = {
        'H': 'herbal',
        'T': 'text',
        'A': 'astronomical',
        'C': 'cosmological',
        'Z': 'zodiac',
        'B': 'balneological',
        'P': 'pharmaceutical',
        'S': 'stars'
    }

    pages = []
    for line in page_lines:
        page_match = re.search(r'<f(\d+[rv]\d*|Ros)>', line)
        illustration_match = re.search(r'\$I=(\w)', line)
        currier_language_match = re.search(r'\$L=(\w)', line)
        scribe_hand_match = re.search(r'\$H=(\w)', line)
        entry = {
            "page": page_match.group(1) if page_match else None,
            "illustration": illustration_map[illustration_match.group(1)] if illustration_match else None,
            "currier_language": currier_language_match.group(1) if currier_language_match else None,
            "scribe_hand": scribe_hand_match.group(1) if scribe_hand_match else None
        }
        pages.append(entry)

    return pages


def extract_locus_data(lines):
    """Extracts data about text loci from the IVTFF file lines."""
    locus_regex = re.compile(r'<f(\d+[rv]\d*|Ros)\.\d+,[@+*=&~/!][PLCR].>')
    locus_lines = [l for l in lines if locus_regex.match(l)]

    loci = []
    for line in locus_lines:
        match = re.search(r'<f(\d+[rv]\d*|Ros)\.(\d+),[@+*=&~/!]([PLCR]).>\s+(.+)', line)
        entry = {
            "page": match.group(1) if match else None,
            "in_page_count": match.group(2) if match else None,
            "locus_type": match.group(3) if match else None,
            "text": match.group(4) if match else None,
        }
        loci.append(entry)

    return loci


def generate_paragraphs(loci):
    """Stitches together paragraphs from the text at multiple loci."""
    paragraphs = []
    para_text = ''
    page_items = []
    for locus in [l for l in loci if l['locus_type'] in ['P', 'C']]:
        text = locus['text']
        if text[:3] == '<%>':
            para_text = text[3:]
            page_items = [locus['in_page_count']]
        elif text[-3:] == '<$>':
            para_text += '.' + text[:-3]
            page_items.append(locus['in_page_count'])
            paragraphs.append({
                'page': locus['page'],
                'page_items': page_items,
                'locus_type': locus['locus_type'],
                'text': para_text,
                'text_len': len(para_text)})
            para_text = ''
            page_items = []
        elif para_text == '' and len(page_items) == 0:
            paragraphs.append({
                'page': locus['page'],
                'page_items': [locus['in_page_count']],
                'locus_type': locus['locus_type'],
                'text': text,
                'text_len': len(text)})
        else:
            para_text += '.' + text
            page_items.append(locus['in_page_count'])
    return paragraphs


def generate_page_texts(
    paragraphs: list[dict], 
    pages: list[dict],
    tokenizer: LlamaTokenizer
) -> pd.DataFrame:
    """
    Generates a DataFrame with an entry per page, with the text from all paragraphs 
    on that page.
    Only paragraph loci are included, not circular, radial or label loci.
    This means that some pages (those without paragraphs) are not included.
    A DataFrame is returned since this is easier for subsequent steps to work with.
    """
    paragraphs_df = pd.DataFrame(paragraphs)
    pages_df = pd.DataFrame(pages)
    page_text_df = paragraphs_df[
        paragraphs_df['locus_type'] == 'P'].groupby('page')['text'].apply('|'.join).reset_index()
    page_text_df['text_len'] = page_text_df['text'].apply(len)
    page_text_df = page_text_df.merge(
        pages_df, how='left', left_on='page', right_on='page')
    page_text_df['spaced_text'] = page_text_df['text'].apply(
        lambda x: x.replace('.', ' ').replace(',', ' ').replace('|', ' | '))
    page_text_df['token_count'] = page_text_df['text'].apply(
        lambda x: len(tokenizer.encode(x)))
    page_text_df['token_count_spaced'] = page_text_df['spaced_text'].apply(
        lambda x: len(tokenizer.encode(x)))
    return page_text_df


def train_val_split_for_category(
    page_text_df: pd.DataFrame,
    category: tuple
) -> tuple[list[str], list[str]]:
    """
    Split the DataFrame into training and validation sets for a specific category.
    """
    illustration, currier_language = category
    category_df = page_text_df[
        (page_text_df['illustration'] == illustration) & 
        (page_text_df['currier_language'] == currier_language)
    ].sort_values(
        'token_count_spaced', ascending=False)
    
    texts = list(category_df['spaced_text'])
    pages = list(category_df['page'])
    
    val_indices = list(range(2, len(category_df), 5))
    train_indices = [i for i in range(len(category_df)) if i not in val_indices]

    train_data = [{'page': pages[i], 'text': texts[i]} for i in train_indices]
    val_data = [{'page': pages[i], 'text': texts[i]} for i in val_indices]

    return train_data, val_data


def create_sliding_windows(
    tokenizer: LlamaTokenizer,
    text: str, 
    window_size: int = WINDOW_SIZE, 
    stride: int = STRIDE
) -> list[str]:
    
    # First tokenize to get proper boundaries
    tokens = tokenizer.encode(text)
    
    windows = []
    for i in range(0, len(tokens), stride):
        window_tokens = tokens[i:i + window_size]
        if len(window_tokens) >= 50:  # Minimum window size
            window_text = tokenizer.decode(window_tokens)
            windows.append(window_text)
    
    return windows


def generate_datasets(
    manuscript_file: str,
    tokenizer: LlamaTokenizer
) -> tuple[Dataset, Dataset]:
    """
    End-to-end generation of HF Datasets from IVTFF file.
    """
    logger.info("Reading manuscript file", manuscript_file=manuscript_file)
    with open(f'voynich_transliterations/{manuscript_file}') as f:
        lines = f.readlines()

    logger.info("Extract page and locus data from manuscript file")
    lines = [l.strip() for l in lines if l[0]!='#'] # Remove comments
    pages = extract_page_data(lines)
    loci = extract_locus_data(lines)

    logger.info("Generating paragraphs and page texts")
    paragraphs = generate_paragraphs(loci)
    page_text_df = generate_page_texts(paragraphs, pages, tokenizer)

    logger.info("Making train/val split")
    groups = page_text_df.groupby(['illustration', 'currier_language'])
    categories = list(groups.count().index)
    train_data = []
    val_data = []
    for category in categories:
        train, val = train_val_split_for_category(page_text_df, category)
        train_data.extend(train)
        val_data.extend(val)
    logger.info("Train/val split complete", train_count=len(train_data), val_count=len(val_data))

    logger.info("Creating sliding windows for training and validation data")
    train_windows = []
    for data in train_data:
        windows = create_sliding_windows(tokenizer, data['text'])
        train_windows.extend([{'page': data['page'], 'window': w} for w in windows])
    val_windows = []
    for data in val_data:
        windows = create_sliding_windows(tokenizer, data['text'])
        val_windows.extend([{'page': data['page'], 'window': w} for w in windows])

    logger.info("Generating HF Datasets from sliding windows")
    train_dataset = Dataset.from_list(train_windows)
    val_dataset = Dataset.from_list(val_windows)

    def tokenise_window(items):
        return tokenizer(
            items['window'],
            truncation=True,
            padding='max_length',
            max_length=WINDOW_SIZE,
            return_tensors='pt'
        )

    train_dataset = train_dataset.map(tokenise_window, batched=True)
    val_dataset = val_dataset.map(tokenise_window, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    return train_dataset, val_dataset