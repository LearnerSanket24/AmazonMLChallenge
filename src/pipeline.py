"""Main end-to-end pipeline for Amazon ML Challenge 2026."""

import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
import sys

from src.utils.io import (
    load_all_sources, write_candidate_pairs, write_matching_results,
    read_source_file, read_ground_truth
)
from src.blocking.blocking import HybridBlocker, run_blocking
from src.features.feature_engineering import FeatureExtractor, build_feature_matrix
from src.matching.matching import (
    EnsembleMatcher, generate_training_labels, evaluate_predictions,
    macro_f05_score, apply_threshold_per_entity
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EntityResolutionPipeline:
    """Complete end-to-end entity resolution pipeline."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.blocker = None
        self.feature_extractor = None
        self.matcher = None
        self.feature_names = []
    
    def train(self, data_dir: str) -> Dict[str, float]:
        """Train the complete pipeline on training data."""
        logger.info("=" * 60)
        logger.info("Starting training pipeline")
        logger.info("=" * 60)
        
        # Load training data
        sources = load_all_sources(data_dir, 'train')
        source1 = sources['source1']
        source2 = sources['source2']
        source3 = sources['source3']
        ground_truth = sources['ground_truth']
        
        logger.info(f"Source 1: {len(source1)} records")
        logger.info(f"Source 2: {len(source2)} records")
        logger.info(f"Source 3: {len(source3)} records")
        logger.info(f"Ground truth: {len(ground_truth)} entities")
        
        # Stage 1: Blocking
        logger.info("\n--- Stage 1: Blocking ---")
        self.blocker = HybridBlocker(self.config)
        self.blocker.build(source1, source2, source3)
        candidate_pairs = self.blocker.generate_candidates()
        
        # Save candidate pairs for audit
        candidate_path = Path(self.config['output']['candidate_pairs'])
        write_candidate_pairs(candidate_pairs, str(candidate_path))
        logger.info(f"Saved {len(candidate_pairs)} candidate pairs to {candidate_path}")
        
        # Generate labels
        logger.info("Generating training labels...")
        y = generate_training_labels(candidate_pairs, ground_truth)
        logger.info(f"Positive pairs: {y.sum()} / {len(y)} ({y.mean()*100:.2f}%)")
        
        # Stage 2: Feature Engineering
        logger.info("\n--- Stage 2: Feature Engineering ---")
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_extractor.fit_tfidf(source1, source2, source3)
        
        X, self.feature_names, valid_pairs = build_feature_matrix(
            candidate_pairs, source1, source2, source3, self.feature_extractor
        )
        
        # Align labels with valid pairs
        # (some pairs might be dropped due to missing records)
        pair_to_label = {pair: label for pair, label in zip(candidate_pairs, y)}
        y_aligned = np.array([pair_to_label[p] for p in valid_pairs])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Aligned labels - Positive: {y_aligned.sum()} / {len(y_aligned)}")
        
        # Train matching model
        logger.info("\n--- Stage 3: Matching Model ---")
        self.matcher = EnsembleMatcher(self.config)
        self.matcher.fit(X, y_aligned, self.feature_names)
        
        # Evaluate on training data (for reference)
        y_proba = self.matcher.predict_proba(X)
        y_pred = self.matcher.predict(X)
        metrics = evaluate_predictions(y_aligned, y_pred, y_proba)
        
        logger.info(f"Training metrics: {metrics}")
        
        # Save models
        self.save_models()
        
        return metrics
    
    def predict(self, data_dir: str, output_path: str) -> Dict[str, List[str]]:
        """Run inference on test data."""
        logger.info("=" * 60)
        logger.info("Starting inference pipeline")
        logger.info("=" * 60)
        
        # Load test data
        sources = load_all_sources(data_dir, 'test')
        source1 = sources['source1']
        source2 = sources['source2']
        source3 = sources['source3']
        
        logger.info(f"Source 1 (test): {len(source1)} records")
        logger.info(f"Source 2 (test): {len(source2)} records")
        logger.info(f"Source 3 (test): {len(source3)} records")
        
        # Stage 1: Blocking
        logger.info("\n--- Stage 1: Blocking ---")
        if self.blocker is None:
            self.blocker = HybridBlocker(self.config)
            self.blocker.build(source1, source2, source3)
        
        candidate_pairs = self.blocker.generate_candidates()
        
        # Save candidate pairs
        candidate_path = Path(self.config['output']['candidate_pairs'])
        write_candidate_pairs(candidate_pairs, str(candidate_path))
        logger.info(f"Saved {len(candidate_pairs)} candidate pairs to {candidate_path}")
        
        # Stage 2: Feature Engineering
        logger.info("\n--- Stage 2: Feature Engineering ---")
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(self.config)
            # Need to fit TF-IDF on test data too (or use training vocab)
            self.feature_extractor.fit_tfidf(source1, source2, source3)
        
        X, _, valid_pairs = build_feature_matrix(
            candidate_pairs, source1, source2, source3, self.feature_extractor
        )
        
        logger.info(f"Test feature matrix shape: {X.shape}")
        
        # Stage 3: Matching
        logger.info("\n--- Stage 3: Matching ---")
        if self.matcher is None:
            raise ValueError("Matcher not trained. Call train() first or load models.")
        
        # Predict probabilities
        y_proba = self.matcher.predict_proba(X)
        
        # Apply threshold per entity
        source1_ids = source1['id'].tolist()
        predictions = apply_threshold_per_entity(
            valid_pairs, y_proba, self.matcher.threshold, source1_ids
        )
        
        # Ensure all source1 entities have entries (singletons get empty list)
        for s1_id in source1_ids:
            if s1_id not in predictions:
                predictions[s1_id] = []
        
        # Save results
        write_matching_results(predictions, output_path)
        logger.info(f"Saved predictions to {output_path}")
        
        return predictions
    
    def save_models(self, model_dir: str = "models") -> None:
        """Save all trained models."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save blocker
        if self.blocker:
            self.blocker.save_index(str(Path(model_dir) / 'blocking_index'))
        
        # Save feature extractor (TF-IDF vectorizers)
        import pickle
        with open(Path(model_dir) / 'feature_extractor.pkl', 'wb') as f:
            pickle.dump({
                'name_tfidf': self.feature_extractor.name_tfidf,
                'address_tfidf': self.feature_extractor.address_tfidf,
                'name_vocab': self.feature_extractor.name_vocab,
                'address_vocab': self.feature_extractor.address_vocab
            }, f)
        
        # Save matcher
        if self.matcher:
            self.matcher.save(str(Path(model_dir) / 'matcher'))
        
        logger.info(f"Saved all models to {model_dir}")
    
    def load_models(self, model_dir: str = "models") -> None:
        """Load trained models."""
        # Load blocker
        self.blocker = HybridBlocker(self.config)
        self.blocker.load_index(str(Path(model_dir) / 'blocking_index'))
        
        # Load feature extractor
        import pickle
        with open(Path(model_dir) / 'feature_extractor.pkl', 'rb') as f:
            fe_data = pickle.load(f)
        
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_extractor.name_tfidf = fe_data['name_tfidf']
        self.feature_extractor.address_tfidf = fe_data['address_tfidf']
        self.feature_extractor.name_vocab = fe_data['name_vocab']
        self.feature_extractor.address_vocab = fe_data['address_vocab']
        
        # Load matcher
        self.matcher = EnsembleMatcher.load(str(Path(model_dir) / 'matcher'), self.config)
        
        logger.info(f"Loaded all models from {model_dir}")


def validate_submission(submission_path: str, ground_truth_path: Optional[str] = None) -> bool:
    """Validate submission format."""
    try:
        df = pd.read_csv(submission_path, sep='\t', dtype=str, keep_default_na=False)
        
        # Check columns
        if list(df.columns) != ['source1_id', 'matched_ids']:
            logger.error(f"Invalid columns: {list(df.columns)}. Expected: ['source1_id', 'matched_ids']")
            return False
        
        # Check for duplicates
        if df['source1_id'].duplicated().any():
            logger.error("Duplicate source1_id found")
            return False
        
        # Check matched_ids format
        for _, row in df.iterrows():
            matched = row['matched_ids']
            if matched and matched != '':
                ids = matched.split(',')
                if any(not id.strip() for id in ids):
                    logger.error(f"Empty ID in matched_ids: {matched}")
                    return False
        
        logger.info(f"Submission validation passed: {len(df)} rows")
        
        # If ground truth provided, compute score
        if ground_truth_path and Path(ground_truth_path).exists():
            gt = read_ground_truth(ground_truth_path)
            pred_dict = dict(zip(df['source1_id'], df['matched_ids'].apply(
                lambda x: x.split(',') if x else []
            )))
            score = macro_f05_score(pred_dict, gt)
            logger.info(f"Macro F0.5 Score: {score:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Amazon ML Challenge 2026 - Entity Resolution Pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['train', 'predict', 'validate'], required=True, help='Mode')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output', default='submission/matching_results.tsv', help='Output path')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--ground-truth', help='Ground truth path for validation')
    
    args = parser.parse_args()
    
    pipeline = EntityResolutionPipeline(args.config)
    
    if args.mode == 'train':
        pipeline.train(args.data_dir)
        pipeline.save_models(args.model_dir)
    
    elif args.mode == 'predict':
        pipeline.load_models(args.model_dir)
        pipeline.predict(args.data_dir, args.output)
    
    elif args.mode == 'validate':
        validate_submission(args.output, args.ground_truth)


if __name__ == '__main__':
    main()