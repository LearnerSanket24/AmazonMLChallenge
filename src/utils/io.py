"""I/O utilities for reading/writing TSV files with explicit tab separation."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def read_tsv(filepath: str, **kwargs) -> pd.DataFrame:
    """Read TSV file with explicit tab separator."""
    default_kwargs = {
        'sep': '\t',
        'dtype': str,
        'keep_default_na': False,
        'na_filter': False
    }
    default_kwargs.update(kwargs)
    return pd.read_csv(filepath, **default_kwargs)


def write_tsv(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    """Write DataFrame to TSV with explicit tab separator."""
    default_kwargs = {
        'sep': '\t',
        'index': False,
        'header': True
    }
    default_kwargs.update(kwargs)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, **default_kwargs)
    logger.info(f"Written {len(df)} rows to {filepath}")


def read_source_file(filepath: str) -> pd.DataFrame:
    """Read source file (source1, source2, source3) with expected columns.
    
    Handles both canonical column names (id, name, address) and the actual
    dataset column names (entity_id, business_name, business_address).
    """
    df = read_tsv(filepath)

    # Remap actual dataset column names to the canonical names used throughout the pipeline
    rename_map = {
        'entity_id': 'id',
        'business_name': 'name',
        'business_address': 'address',
    }
    df = df.rename(columns=rename_map)

    expected_cols = ['id', 'name', 'address']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Source file must have columns: {expected_cols}. Found: {list(df.columns)}")
    return df[expected_cols].copy()


def read_ground_truth(filepath: str) -> pd.DataFrame:
    """Read ground truth file with source1_id and matched_ids columns.

    Handles both canonical column names (source1_id, matched_ids) and the actual
    dataset column names (source1_entity_id, matched_entity_ids).
    """
    df = read_tsv(filepath)

    # Remap actual dataset column names to the canonical names used throughout the pipeline
    rename_map = {
        'source1_entity_id': 'source1_id',
        'matched_entity_ids': 'matched_ids',
    }
    df = df.rename(columns=rename_map)

    expected_cols = ['source1_id', 'matched_ids']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Ground truth must have columns: {expected_cols}. Found: {list(df.columns)}")
    return df[expected_cols].copy()


def write_candidate_pairs(pairs: List[Tuple[str, str, str]], filepath: str) -> None:
    """Write candidate pairs to TSV.
    
    Args:
        pairs: List of (source1_id, source_id, source_name) tuples
               where source_name is 'source2' or 'source3'
        filepath: Output path
    """
    df = pd.DataFrame(pairs, columns=['source1_id', 'candidate_id', 'source'])
    write_tsv(df, filepath)


def write_matching_results(results: Dict[str, List[str]], filepath: str) -> None:
    """Write matching results in submission format.
    
    Args:
        results: Dict mapping source1_id -> list of matched candidate_ids
        filepath: Output path
    """
    rows = []
    for source1_id, matched_ids in results.items():
        matched_str = ','.join(matched_ids) if matched_ids else ''
        rows.append({'source1_id': source1_id, 'matched_ids': matched_str})
    
    df = pd.DataFrame(rows, columns=['source1_id', 'matched_ids'])
    write_tsv(df, filepath)


def load_all_sources(data_dir: str, split: str = 'train') -> Dict[str, pd.DataFrame]:
    """Load all source files for a given split.

    Tries two filename conventions:
      1. Canonical (old): ``source{i}_{split}.tsv``  e.g. ``source1_train.tsv``
      2. Actual dataset:  ``{split}_source{i}.tsv``  e.g. ``train_source1.tsv``
    """
    sources = {}
    for i in [1, 2, 3]:
        # Try both naming conventions
        candidates = [
            Path(data_dir) / f"source{i}_{split}.tsv",
            Path(data_dir) / f"{split}_source{i}.tsv",
        ]
        filepath = next((p for p in candidates if p.exists()), None)
        if filepath:
            sources[f'source{i}'] = read_source_file(str(filepath))
            logger.info(f"Loaded {len(sources[f'source{i}'])} records from {filepath.name}")
        else:
            logger.warning(f"File not found for source{i}/{split}. Tried: {[str(p) for p in candidates]}")

    if split == 'train':
        gt_candidates = [
            Path(data_dir) / "ground_truth_train.tsv",
            Path(data_dir) / "train_ground_truth.tsv",
        ]
        gt_path = next((p for p in gt_candidates if p.exists()), None)
        if gt_path:
            sources['ground_truth'] = read_ground_truth(str(gt_path))
            logger.info(f"Loaded {len(sources['ground_truth'])} ground truth entries from {gt_path.name}")
        else:
            logger.warning(f"Ground truth not found. Tried: {[str(p) for p in gt_candidates]}")

    return sources


def parse_matched_ids(matched_ids_str: str) -> List[str]:
    """Parse comma-separated matched IDs string into list."""
    if not matched_ids_str or matched_ids_str.strip() == '':
        return []
    return [x.strip() for x in matched_ids_str.split(',') if x.strip()]


def format_matched_ids(matched_ids: List[str]) -> str:
    """Format list of matched IDs into comma-separated string."""
    return ','.join(matched_ids) if matched_ids else ''