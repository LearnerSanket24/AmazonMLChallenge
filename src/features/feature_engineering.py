"""Feature engineering for pairwise matching."""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from functools import lru_cache

from src.utils.text_normalization import (
    normalize_name, normalize_address, extract_tokens, 
    jaccard_similarity, token_overlap_ratio, parse_address_components,
    soundex, metaphone, nysiis
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts pairwise similarity features for name and address pairs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_config = config.get('features', {})
        
        # TF-IDF vectorizers (fitted on training data)
        self.name_tfidf = None
        self.address_tfidf = None
        self.name_vocab = None
        self.address_vocab = None
        
        # Embedding models (lazy loaded)
        self.embedding_models = {}
    
    def fit_tfidf(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> None:
        """Fit TF-IDF vectorizers on all names and addresses."""
        logger.info("Fitting TF-IDF vectorizers...")
        
        all_names = []
        all_addresses = []
        
        for df in [source1, source2, source3]:
            all_names.extend(df['name'].fillna('').astype(str).tolist())
            all_addresses.extend(df['address'].fillna('').astype(str).tolist())
        
        # Normalize
        all_names = [normalize_name(n) for n in all_names]
        all_addresses = [normalize_address(a) for a in all_addresses]
        
        # Name TF-IDF (character n-grams)
        self.name_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            max_features=50000,
            sublinear_tf=True
        )
        self.name_tfidf.fit(all_names)
        self.name_vocab = set(self.name_tfidf.get_feature_names_out())
        
        # Address TF-IDF (word n-grams)
        self.address_tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            max_features=30000,
            sublinear_tf=True
        )
        self.address_tfidf.fit(all_addresses)
        self.address_vocab = set(self.address_tfidf.get_feature_names_out())
        
        logger.info(f"Name TF-IDF vocab size: {len(self.name_vocab)}")
        logger.info(f"Address TF-IDF vocab size: {len(self.address_vocab)}")
    
    def _get_embedding_model(self, model_name: str):
        """Lazy load embedding model."""
        if model_name not in self.embedding_models:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except ImportError:
                logger.warning(f"Could not load {model_name}, sentence-transformers not available")
                return None
        return self.embedding_models[model_name]
    
    def extract_name_features(self, name1: str, name2: str) -> Dict[str, float]:
        """Extract name similarity features."""
        features = {}
        
        norm1 = normalize_name(name1)
        norm2 = normalize_name(name2)
        
        if not norm1 or not norm2:
            return {k: 0.0 for k in self._get_name_feature_names()}
        
        tokens1 = extract_tokens(norm1, stopwords=None)
        tokens2 = extract_tokens(norm2, stopwords=None)
        set1, set2 = set(tokens1), set(tokens2)
        
        # Exact match
        features['name_exact_match'] = float(norm1 == norm2)
        
        # Prefix/Suffix match
        features['name_prefix_match'] = float(norm1.split()[0] == norm2.split()[0]) if tokens1 and tokens2 else 0.0
        features['name_suffix_match'] = float(norm1.split()[-1] == norm2.split()[-1]) if tokens1 and tokens2 else 0.0
        
        # Token overlap
        features['name_token_jaccard'] = jaccard_similarity(set1, set2)
        features['name_token_overlap_ratio'] = token_overlap_ratio(tokens1, tokens2)
        features['name_token_count_ratio'] = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0.0
        
        # Common tokens
        common = set1 & set2
        features['name_common_token_count'] = float(len(common))
        features['name_unique_token_count'] = float(len(set1 ^ set2))
        
        # Abbreviation detection
        features['name_abbreviation_match'] = self._check_abbreviation_match(norm1, norm2)
        
        # Phonetic matching
        features['name_soundex_match'] = float(soundex(norm1) == soundex(norm2))
        features['name_metaphone_match'] = float(metaphone(norm1) == metaphone(norm2))
        features['name_nysiis_match'] = float(nysiis(norm1) == nysiis(norm2))
        
        # String similarity metrics
        features['name_jaro_winkler'] = self._jaro_winkler(norm1, norm2)
        features['name_levenshtein_norm'] = 1.0 - self._normalized_levenshtein(norm1, norm2)
        
        # TF-IDF cosine similarity
        if self.name_tfidf is not None:
            try:
                vec1 = self.name_tfidf.transform([norm1])
                vec2 = self.name_tfidf.transform([norm2])
                features['name_tfidf_cosine'] = float(cosine_similarity(vec1, vec2)[0, 0])
            except:
                features['name_tfidf_cosine'] = 0.0
        else:
            features['name_tfidf_cosine'] = 0.0
        
        # Embedding similarity
        for model_name in self.features_config.get('embedding_models', []):
            model = self._get_embedding_model(model_name)
            if model is not None:
                try:
                    emb1 = model.encode([norm1], normalize_embeddings=True)
                    emb2 = model.encode([norm2], normalize_embeddings=True)
                    sim = float(np.dot(emb1[0], emb2[0]))
                    model_short = model_name.split('/')[-1].replace('-', '_')
                    features[f'name_emb_{model_short}_cosine'] = sim
                except:
                    features[f'name_emb_{model_name.split("/")[-1]}_cosine'] = 0.0
        
        return features
    
    def extract_address_features(self, addr1: str, addr2: str) -> Dict[str, float]:
        """Extract address similarity features."""
        features = {}
        
        norm1 = normalize_address(addr1)
        norm2 = normalize_address(addr2)
        
        if not norm1 or not norm2:
            return {k: 0.0 for k in self._get_address_feature_names()}
        
        comp1 = parse_address_components(norm1)
        comp2 = parse_address_components(norm2)
        
        # Exact match
        features['addr_exact_match'] = float(norm1 == norm2)
        
        # House number match
        features['addr_house_num_match'] = float(comp1['house_number'] == comp2['house_number'] and comp1['house_number'] != '')
        
        # Street similarity
        street_tokens1 = extract_tokens(comp1['street'], stopwords=None)
        street_tokens2 = extract_tokens(comp2['street'], stopwords=None)
        features['addr_street_jaccard'] = jaccard_similarity(set(street_tokens1), set(street_tokens2))
        features['addr_street_overlap'] = token_overlap_ratio(street_tokens1, street_tokens2)
        
        # City match
        features['addr_city_match'] = float(comp1['city'].lower() == comp2['city'].lower() and comp1['city'] != '')
        
        # State match
        features['addr_state_match'] = float(comp1['state'].lower() == comp2['state'].lower() and comp1['state'] != '')
        
        # Postal code match
        features['addr_postal_match'] = float(comp1['postal_code'] == comp2['postal_code'] and comp1['postal_code'] != '')
        features['addr_postal_prefix_match'] = float(
            comp1['postal_code'][:3] == comp2['postal_code'][:3] 
            and len(comp1['postal_code']) >= 3 and len(comp2['postal_code']) >= 3
        )
        
        # Full address token overlap
        tokens1 = extract_tokens(norm1, stopwords=None)
        tokens2 = extract_tokens(norm2, stopwords=None)
        features['addr_token_jaccard'] = jaccard_similarity(set(tokens1), set(tokens2))
        features['addr_token_overlap'] = token_overlap_ratio(tokens1, tokens2)
        
        # Landmark overlap (non-standard address tokens)
        landmark_tokens1 = self._extract_landmark_tokens(norm1)
        landmark_tokens2 = self._extract_landmark_tokens(norm2)
        features['addr_landmark_overlap'] = jaccard_similarity(landmark_tokens1, landmark_tokens2)
        
        # String similarity
        features['addr_jaro_winkler'] = self._jaro_winkler(norm1, norm2)
        features['addr_levenshtein_norm'] = 1.0 - self._normalized_levenshtein(norm1, norm2)
        
        # TF-IDF cosine similarity
        if self.address_tfidf is not None:
            try:
                vec1 = self.address_tfidf.transform([norm1])
                vec2 = self.address_tfidf.transform([norm2])
                features['addr_tfidf_cosine'] = float(cosine_similarity(vec1, vec2)[0, 0])
            except:
                features['addr_tfidf_cosine'] = 0.0
        else:
            features['addr_tfidf_cosine'] = 0.0
        
        # Embedding similarity
        for model_name in self.features_config.get('embedding_models', []):
            model = self._get_embedding_model(model_name)
            if model is not None:
                try:
                    emb1 = model.encode([norm1], normalize_embeddings=True)
                    emb2 = model.encode([norm2], normalize_embeddings=True)
                    sim = float(np.dot(emb1[0], emb2[0]))
                    model_short = model_name.split('/')[-1].replace('-', '_')
                    features[f'addr_emb_{model_short}_cosine'] = sim
                except:
                    features[f'addr_emb_{model_name.split("/")[-1]}_cosine'] = 0.0
        
        return features
    
    def extract_cross_features(self, name1: str, addr1: str, name2: str, addr2: str) -> Dict[str, float]:
        """Extract cross-field features (name-address consistency)."""
        features = {}
        
        norm_name1 = normalize_name(name1)
        norm_addr1 = normalize_address(addr1)
        norm_name2 = normalize_name(name2)
        norm_addr2 = normalize_address(addr2)
        
        # Shared tokens between name1 and addr2 (and vice versa)
        name1_tokens = set(extract_tokens(norm_name1, stopwords=None))
        addr1_tokens = set(extract_tokens(norm_addr1, stopwords=None))
        name2_tokens = set(extract_tokens(norm_name2, stopwords=None))
        addr2_tokens = set(extract_tokens(norm_addr2, stopwords=None))
        
        features['cross_name1_addr2_overlap'] = jaccard_similarity(name1_tokens, addr2_tokens)
        features['cross_name2_addr1_overlap'] = jaccard_similarity(name2_tokens, addr1_tokens)
        features['cross_name_name_overlap'] = jaccard_similarity(name1_tokens, name2_tokens)
        features['cross_addr_addr_overlap'] = jaccard_similarity(addr1_tokens, addr2_tokens)
        
        # Consistency: if names match, addresses should match
        name_sim = self._jaro_winkler(norm_name1, norm_name2)
        addr_sim = self._jaro_winkler(norm_addr1, norm_addr2)
        features['cross_name_addr_consistency'] = 1.0 - abs(name_sim - addr_sim)
        
        return features
    
    def extract_all_features(self, name1: str, addr1: str, name2: str, addr2: str) -> Dict[str, float]:
        """Extract all features for a pair."""
        features = {}
        features.update(self.extract_name_features(name1, name2))
        features.update(self.extract_address_features(addr1, addr2))
        features.update(self.extract_cross_features(name1, addr1, name2, addr2))
        return features
    
    def _get_name_feature_names(self) -> List[str]:
        """Get list of name feature names."""
        base = [
            'name_exact_match', 'name_prefix_match', 'name_suffix_match',
            'name_token_jaccard', 'name_token_overlap_ratio', 'name_token_count_ratio',
            'name_common_token_count', 'name_unique_token_count',
            'name_abbreviation_match', 'name_soundex_match', 'name_metaphone_match',
            'name_nysiis_match', 'name_jaro_winkler', 'name_levenshtein_norm',
            'name_tfidf_cosine'
        ]
        # Add embedding features
        for model_name in self.features_config.get('embedding_models', []):
            model_short = model_name.split('/')[-1].replace('-', '_')
            base.append(f'name_emb_{model_short}_cosine')
        return base
    
    def _get_address_feature_names(self) -> List[str]:
        """Get list of address feature names."""
        base = [
            'addr_exact_match', 'addr_house_num_match', 'addr_street_jaccard',
            'addr_street_overlap', 'addr_city_match', 'addr_state_match',
            'addr_postal_match', 'addr_postal_prefix_match',
            'addr_token_jaccard', 'addr_token_overlap', 'addr_landmark_overlap',
            'addr_jaro_winkler', 'addr_levenshtein_norm', 'addr_tfidf_cosine'
        ]
        for model_name in self.features_config.get('embedding_models', []):
            model_short = model_name.split('/')[-1].replace('-', '_')
            base.append(f'addr_emb_{model_short}_cosine')
        return base
    
    def _check_abbreviation_match(self, name1: str, name2: str) -> float:
        """Check if one name is abbreviation of another."""
        tokens1 = name1.split()
        tokens2 = name2.split()
        
        if len(tokens1) != len(tokens2):
            return 0.0
        
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2:
                continue
            # Check if one is abbreviation of other
            if len(t1) == 1 and t2.startswith(t1):
                continue
            if len(t2) == 1 and t1.startswith(t2):
                continue
            return 0.0
        
        return 1.0
    
    def _extract_landmark_tokens(self, address: str) -> Set[str]:
        """Extract potential landmark tokens from address."""
        # Tokens that might be landmarks (not standard address components)
        standard = {'st', 'ave', 'rd', 'dr', 'ln', 'ct', 'blvd', 'pl', 'cir', 'way',
                    'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw', 'ste', 'apt', 'unit',
                    'floor', 'fl', 'rm', 'bldg', 'building', 'nr', 'near', 'opp', 'opposite'}
        
        tokens = set(extract_tokens(address, stopwords=None))
        return tokens - standard
    
    def _jaro_winkler(self, s1: str, s2: str, p: float = 0.1) -> float:
        """Jaro-Winkler similarity."""
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Jaro distance
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0
        
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        
        # Winkler adjustment
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break
        
        return jaro + (prefix * p * (1 - jaro))
    
    def _normalized_levenshtein(self, s1: str, s2: str) -> float:
        """Normalized Levenshtein distance."""
        if s1 == s2:
            return 0.0
        if not s1 or not s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        dp = list(range(len2 + 1))
        
        for i in range(1, len1 + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, len2 + 1):
                curr = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = curr
        
        return dp[len2] / max(len1, len2)


def build_feature_matrix(
    candidate_pairs: List[Tuple[str, str, str]],
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    feature_extractor: FeatureExtractor
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str, str]]]:
    """Build feature matrix for all candidate pairs."""
    logger.info(f"Building feature matrix for {len(candidate_pairs)} pairs...")
    
    # Create lookup dictionaries
    source1_dict = source1.set_index('id').to_dict('index')
    source2_dict = source2.set_index('id').to_dict('index')
    source3_dict = source3.set_index('id').to_dict('index')
    
    source23_dict = {**{('source2', k): v for k, v in source2_dict.items()},
                     **{('source3', k): v for k, v in source3_dict.items()}}
    
    feature_rows = []
    valid_pairs = []
    
    for i, (s1_id, cand_id, source) in enumerate(candidate_pairs):
        if i % 10000 == 0:
            logger.info(f"  Processed {i}/{len(candidate_pairs)} pairs")
        
        s1_record = source1_dict.get(s1_id)
        s23_record = source23_dict.get((source, cand_id))
        
        if s1_record is None or s23_record is None:
            continue
        
        features = feature_extractor.extract_all_features(
            s1_record['name'], s1_record['address'],
            s23_record['name'], s23_record['address']
        )
        
        feature_rows.append(features)
        valid_pairs.append((s1_id, cand_id, source))
    
    if not feature_rows:
        return np.array([]), [], []
    
    # Convert to DataFrame then numpy
    feature_df = pd.DataFrame(feature_rows)
    feature_names = list(feature_df.columns)
    
    # Handle missing values
    feature_df = feature_df.fillna(0.0)
    
    # Replace inf values
    feature_df = feature_df.replace([np.inf, -np.inf], 0.0)
    
    return feature_df.values.astype(np.float32), feature_names, valid_pairs