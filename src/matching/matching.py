"""Matching models for pairwise classification."""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from sklearn.calibration import CalibratedClassifierCV
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import torch
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

from src.utils.io import read_ground_truth

logger = logging.getLogger(__name__)


class MatchingModel:
    """Base class for matching models."""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'MatchingModel':
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'MatchingModel':
        with open(path, 'rb') as f:
            return pickle.load(f)


class XGBoostModel(MatchingModel):
    """XGBoost classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'XGBoostModel':
        if not HAS_XGB:
            raise ImportError("XGBoost not installed")
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)


class LightGBMModel(MatchingModel):
    """LightGBM classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'LightGBMModel':
        if not HAS_LGB:
            raise ImportError("LightGBM not installed")
        
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)


class CatBoostModel(MatchingModel):
    """CatBoost classifier."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'CatBoostModel':
        if not HAS_CAT:
            raise ImportError("CatBoost not installed")
        
        self.model = cb.CatBoostClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight, verbose=False)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)


class CrossEncoderModel:
    """Cross-encoder for re-ranking (uses transformer)."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 batch_size: int = 32, max_length: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
    
    def _load_model(self):
        if self.model is None and HAS_CROSS_ENCODER:
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
    
    def predict_proba(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Predict probabilities for text pairs.
        
        Args:
            pairs: List of (text1, text2) tuples
        Returns:
            Array of shape (n_pairs, 2) with probabilities
        """
        self._load_model()
        if self.model is None:
            return np.zeros((len(pairs), 2))
        
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True)
        # Convert scores to probabilities (sigmoid)
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        probs = np.column_stack([1 - probs_pos, probs_pos])
        return probs
    
    def save(self, path: str) -> None:
        pass  # CrossEncoder doesn't need saving
    
    @classmethod
    def load(cls, path: str) -> 'CrossEncoderModel':
        return cls()


class EnsembleMatcher:
    """Ensemble of multiple matching models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.matching_config = config.get('matching', {})
        self.models: Dict[str, MatchingModel] = {}
        self.cross_encoders: Dict[str, CrossEncoderModel] = {}
        self.weights = self.matching_config.get('ensemble_weights', {})
        self.threshold = 0.5
        self.feature_names = []
    
    def add_model(self, name: str, model: MatchingModel) -> None:
        self.models[name] = model
    
    def add_cross_encoder(self, name: str, model: CrossEncoderModel) -> None:
        self.cross_encoders[name] = model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str],
            sample_weight: Optional[np.ndarray] = None) -> 'EnsembleMatcher':
        """Train all models in ensemble."""
        self.feature_names = feature_names
        
        logger.info("Training ensemble models...")
        
        # Train tree-based models
        for model_name, model_config in self.matching_config.get('models', []):
            if model_name == 'xgboost' and HAS_XGB:
                logger.info("Training XGBoost...")
                model = XGBoostModel(model_name, model_config.get('params', {}))
                model.fit(X, y, sample_weight)
                self.add_model(model_name, model)
            
            elif model_name == 'lightgbm' and HAS_LGB:
                logger.info("Training LightGBM...")
                model = LightGBMModel(model_name, model_config.get('params', {}))
                model.fit(X, y, sample_weight)
                self.add_model(model_name, model)
            
            elif model_name == 'catboost' and HAS_CAT:
                logger.info("Training CatBoost...")
                model = CatBoostModel(model_name, model_config.get('params', {}))
                model.fit(X, y, sample_weight)
                self.add_model(model_name, model)
        
        # Initialize cross-encoders
        for ce_name in self.matching_config.get('cross_encoders', []):
            logger.info(f"Initializing cross-encoder: {ce_name}")
            self.add_cross_encoder(ce_name, CrossEncoderModel(ce_name))
        
        # Optimize threshold on validation set
        self.optimize_threshold(X, y)
        
        return self
    
    def optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal threshold for F0.5 score."""
        logger.info("Optimizing threshold for F0.5...")
        
        # Get ensemble predictions
        y_proba = self.predict_proba(X)
        
        search_config = self.matching_config.get('training', {}).get('threshold_search', {})
        min_thresh = search_config.get('min_threshold', 0.1)
        max_thresh = search_config.get('max_threshold', 0.9)
        steps = search_config.get('steps', 81)
        
        thresholds = np.linspace(min_thresh, max_thresh, steps)
        best_f05 = -1
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_proba[:, 1] >= thresh).astype(int)
            f05 = fbeta_score(y, y_pred, beta=0.5, average='binary', zero_division=0)
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = thresh
        
        self.threshold = best_thresh
        logger.info(f"Optimal threshold: {best_thresh:.4f} (F0.5: {best_f05:.4f})")
        return best_thresh
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble predictions."""
        if not self.models:
            raise ValueError("No models trained")
        
        all_probas = []
        total_weight = 0
        
        # Tree-based models
        for name, model in self.models.items():
            if model.is_fitted:
                weight = self.weights.get(name, 1.0)
                probs = model.predict_proba(X)
                all_probas.append(probs * weight)
                total_weight += weight
        
        # Cross-encoders (would need text pairs, skip for now)
        # In practice, cross-encoders are applied as re-ranking on top candidates
        
        if not all_probas:
            raise ValueError("No fitted models available")
        
        ensemble_proba = np.sum(all_probas, axis=0) / total_weight
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict with optimized threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probas = self.predict_proba(X)
        return (probas[:, 1] >= thresh).astype(int)
    
    def save(self, path: str) -> None:
        """Save ensemble to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model.save(Path(path) / f'{name}_model.pkl')
        
        # Save ensemble metadata
        meta = {
            'weights': self.weights,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'model_names': list(self.models.keys()),
            'cross_encoder_names': list(self.cross_encoders.keys())
        }
        with open(Path(path) / 'ensemble_meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved ensemble to {path}")
    
    @classmethod
    def load(cls, path: str, config: Dict[str, Any]) -> 'EnsembleMatcher':
        """Load ensemble from disk."""
        with open(Path(path) / 'ensemble_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        
        ensemble = cls(config)
        ensemble.weights = meta['weights']
        ensemble.threshold = meta['threshold']
        ensemble.feature_names = meta['feature_names']
        
        for name in meta['model_names']:
            model = MatchingModel.load(Path(path) / f'{name}_model.pkl')
            ensemble.add_model(name, model)
        
        for name in meta['cross_encoder_names']:
            ensemble.add_cross_encoder(name, CrossEncoderModel(name))
        
        logger.info(f"Loaded ensemble from {path}")
        return ensemble


def create_ground_truth_pairs(ground_truth: pd.DataFrame) -> Dict[str, set]:
    """Convert ground truth to lookup dictionary."""
    gt_dict = {}
    for _, row in ground_truth.iterrows():
        s1_id = row['source1_id']
        matched = row['matched_ids']
        if isinstance(matched, str) and matched:
            gt_dict[s1_id] = set(matched.split(','))
        else:
            gt_dict[s1_id] = set()
    return gt_dict


def generate_training_labels(
    candidate_pairs: List[Tuple[str, str, str]],
    ground_truth: pd.DataFrame
) -> np.ndarray:
    """Generate binary labels for candidate pairs from ground truth."""
    gt_dict = create_ground_truth_pairs(ground_truth)
    
    labels = []
    for s1_id, cand_id, source in candidate_pairs:
        true_matches = gt_dict.get(s1_id, set())
        labels.append(1 if cand_id in true_matches else 0)
    
    return np.array(labels, dtype=np.int32)


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Evaluate predictions with multiple metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    f05 = fbeta_score(y_true, y_pred, beta=0.5, average='binary', zero_division=0)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f0.5': f05,
        'support_pos': int(y_true.sum()),
        'support_neg': int((1 - y_true).sum())
    }
    
    if y_proba is not None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
        except:
            pass
    
    return metrics


def macro_f05_score(
    predictions: Dict[str, List[str]], 
    ground_truth: pd.DataFrame
) -> float:
    """Compute macro F0.5 score as per challenge metric."""
    gt_dict = create_ground_truth_pairs(ground_truth)
    
    f05_scores = []
    
    for s1_id, true_matches in gt_dict.items():
        pred_matches = set(predictions.get(s1_id, []))
        
        if not true_matches and not pred_matches:
            # Singleton correctly predicted
            f05_scores.append(1.0)
        elif not true_matches and pred_matches:
            # Singleton with false positives
            f05_scores.append(0.0)
        else:
            # Has true matches
            tp = len(true_matches & pred_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision + recall == 0:
                f05 = 0.0
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            
            f05_scores.append(f05)
    
    return np.mean(f05_scores)


def apply_threshold_per_entity(
    candidate_pairs: List[Tuple[str, str, str]],
    probabilities: np.ndarray,
    threshold: float,
    source1_ids: List[str]
) -> Dict[str, List[str]]:
    """Apply threshold and group predictions by source1 entity."""
    results = {s1_id: [] for s1_id in source1_ids}
    
    for (s1_id, cand_id, source), prob in zip(candidate_pairs, probabilities[:, 1]):
        if prob >= threshold:
            results[s1_id].append(cand_id)
    
    return results