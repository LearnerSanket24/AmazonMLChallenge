"""Amazon ML Challenge 2026 - Entity Resolution Pipeline."""

__version__ = "1.0.0"
__author__ = "Challenge Team"

from src.pipeline import EntityResolutionPipeline
from src.utils.io import (
    read_tsv, write_tsv, read_source_file, read_ground_truth,
    write_candidate_pairs, write_matching_results, load_all_sources
)
from src.blocking.blocking import HybridBlocker, InvertedIndexBlocker, EmbeddingBlocker
from src.features.feature_engineering import FeatureExtractor
from src.matching.matching import (
    EnsembleMatcher, XGBoostModel, LightGBMModel, CatBoostModel,
    CrossEncoderModel, generate_training_labels, macro_f05_score
)

__all__ = [
    "EntityResolutionPipeline",
    "read_tsv", "write_tsv", "read_source_file", "read_ground_truth",
    "write_candidate_pairs", "write_matching_results", "load_all_sources",
    "HybridBlocker", "InvertedIndexBlocker", "EmbeddingBlocker",
    "FeatureExtractor",
    "EnsembleMatcher", "XGBoostModel", "LightGBMModel", "CatBoostModel",
    "CrossEncoderModel", "generate_training_labels", "macro_f05_score",
]