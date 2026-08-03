import os
import re
import logging
import unicodedata
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Configure structured logging
logger = logging.getLogger("WriteLensAI.DatasetLoader")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

SUPPORTED_TEXT_COLS = ['text', 'content', 'sentence', 'document']
SUPPORTED_LABEL_COLS = ['generated', 'label', 'target', 'class']

class DatasetLoader:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

    def find_dataset_file(self, specified_path: str = None) -> str:
        """
        Locates a valid CSV or Excel dataset file.
        Does NOT fabricate synthetic data if missing.
        """
        if specified_path and os.path.exists(specified_path):
            return specified_path

        if os.path.exists(self.dataset_dir):
            for fname in os.listdir(self.dataset_dir):
                if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
                    return os.path.join(self.dataset_dir, fname)

        raise FileNotFoundError(
            f"No labelled dataset file found in '{self.dataset_dir}'. "
            f"Please place a valid CSV or Excel dataset (e.g. ai_vs_human.csv) in '{self.dataset_dir}'."
        )

    def load_and_preprocess(self, dataset_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads CSV/Excel, detects columns, maps labels, cleans text, and generates summary statistics.
        Raises ValueError if dataset is invalid or missing required structures.
        """
        file_path = self.find_dataset_file(dataset_path)
        logger.info(f"Loading dataset from: {file_path}")

        ext = file_path.lower().rsplit('.', 1)[-1]
        try:
            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read dataset file '{file_path}': {str(e)}")

        logger.info(f"Dataset loaded. Initial raw shape: {df.shape[0]} rows, {df.shape[1]} columns.")

        # 1. Column Detection
        text_col = None
        for col in df.columns:
            if col.lower().strip() in SUPPORTED_TEXT_COLS:
                text_col = col
                break

        label_col = None
        for col in df.columns:
            if col.lower().strip() in SUPPORTED_LABEL_COLS:
                label_col = col
                break

        if not text_col:
            raise ValueError(
                f"Text column missing in dataset. Supported column names: {SUPPORTED_TEXT_COLS}. "
                f"Found columns: {list(df.columns)}"
            )

        if not label_col:
            raise ValueError(
                f"Label column missing in dataset. Supported column names: {SUPPORTED_LABEL_COLS}. "
                f"Found columns: {list(df.columns)}"
            )

        logger.info(f"Detected Text Column: '{text_col}', Label Column: '{label_col}'")

        # 2. Label Mapping (0 = Human, 1 = AI)
        raw_labels = df[label_col].dropna().unique()
        label_map = self._infer_label_mapping(raw_labels)
        logger.info(f"Detected Label Mapping: {label_map}")

        # Apply label mapping
        df['target_label'] = df[label_col].map(label_map)
        df = df.dropna(subset=['target_label'])
        df['target_label'] = df['target_label'].astype(int)

        # 3. Data Cleaning
        initial_count = len(df)
        df = df.dropna(subset=[text_col])

        # Unicode normalization & text cleaning
        df['clean_text'] = df[text_col].astype(str).apply(self._clean_raw_text)

        # Remove empty text strings
        df = df[df['clean_text'].str.strip() != '']

        # Remove duplicate text rows
        df = df.drop_duplicates(subset=['clean_text'])

        # Filter texts shorter than 20 words
        df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= 20]

        cleaned_count = len(df)
        logger.info(f"Dataset cleaning complete. Retained {cleaned_count} rows (Removed {initial_count - cleaned_count} duplicates/short/invalid texts).")

        # Validation Checks
        if cleaned_count < 20:
            raise ValueError(f"Dataset contains insufficient valid samples after cleaning ({cleaned_count} samples). Minimum required is 20.")

        unique_classes = df['target_label'].unique()
        if len(unique_classes) < 2:
            raise ValueError(f"Dataset contains only one class ({unique_classes}). Both Human (0) and AI (1) classes are required.")

        # 4. Summary Metrics
        human_count = int((df['target_label'] == 0).sum())
        ai_count = int((df['target_label'] == 1).sum())
        avg_doc_len = float(df['word_count'].mean())

        # Vocabulary size calculation
        vocab = set()
        for t in df['clean_text']:
            vocab.update(re.findall(r'\b\w+\b', t.lower()))
        vocab_size = len(vocab)

        summary = {
            "total_samples": cleaned_count,
            "human_samples": human_count,
            "ai_samples": ai_count,
            "avg_document_length": round(avg_doc_len, 2),
            "vocab_size": vocab_size,
            "class_distribution": {
                "Human (0)": f"{round((human_count / cleaned_count) * 100, 1)}%",
                "AI (1)": f"{round((ai_count / cleaned_count) * 100, 1)}%"
            },
            "text_column": text_col,
            "label_column": label_col,
            "label_mapping": label_map
        }

        self._print_dataset_summary(summary)
        return df[['clean_text', 'target_label', 'word_count']], summary

    def _infer_label_mapping(self, unique_values) -> Dict[Any, int]:
        mapping = {}
        for val in unique_values:
            val_str = str(val).strip().lower()
            if val_str in ['0', 'human', 'human-written', 'original', 'real']:
                mapping[val] = 0
            elif val_str in ['1', 'ai', 'ai-generated', 'generated', 'fake', 'synthetic']:
                mapping[val] = 1
            else:
                # Try numeric conversion fallback
                try:
                    num = int(val)
                    mapping[val] = 1 if num > 0 else 0
                except ValueError:
                    raise ValueError(f"Unable to automatically map label value '{val}' to Human (0) or AI (1).")
        return mapping

    @staticmethod
    def _clean_raw_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        # Trim excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _print_dataset_summary(self, summary: Dict[str, Any]):
        logger.info("=" * 60)
        logger.info("                  DATASET SUMMARY STATISTICS              ")
        logger.info("=" * 60)
        logger.info(f" Total Samples Cleaned:  {summary['total_samples']}")
        logger.info(f" Human Samples (0):      {summary['human_samples']} ({summary['class_distribution']['Human (0)']})")
        logger.info(f" AI Samples (1):         {summary['ai_samples']} ({summary['class_distribution']['AI (1)']})")
        logger.info(f" Avg Document Length:    {summary['avg_document_length']} words")
        logger.info(f" Total Vocabulary Size:  {summary['vocab_size']} unique words")
        logger.info("=" * 60)
