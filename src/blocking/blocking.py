"""Blocking strategies for candidate pair generation."""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path

from src.utils.text_normalization import (
    normalize_name, normalize_address, get_name_blocking_keys, 
    get_address_blocking_keys, extract_postal_code
)
from src.utils.io import read_source_file

logger = logging.getLogger(__name__)


class BlockingKeyGenerator:
    """Generates multiple blocking keys for records."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocking_config = config.get('blocking', {})
    
    def generate_keys(self, name: str, address: str) -> List[str]:
        """Generate all blocking keys for a record."""
        keys = []
        
        # Name-based keys
        name_keys = get_name_blocking_keys(name, self.blocking_config)
        keys.extend(name_keys)
        
        # Address-based keys
        addr_keys = get_address_blocking_keys(address, self.blocking_config)
        keys.extend(addr_keys)
        
        # Cross keys (name + address combinations)
        cross_keys = self._generate_cross_keys(name, address)
        keys.extend(cross_keys)
        
        return keys
    
    def _generate_cross_keys(self, name: str, address: str) -> List[str]:
        """Generate cross-field blocking keys."""
        keys = []
        name_norm = normalize_name(name)
        addr_norm = normalize_address(address)
        
        # First token of name + postal code prefix
        name_tokens = name_norm.split()
        postal = extract_postal_code(addr_norm)
        
        if name_tokens and postal:
            prefix_len = self.blocking_config.get('postal_code_prefix_length', 3)
            keys.append(f"cross:{name_tokens[0]}_{postal[:prefix_len]}")
        
        # First name token + first address token
        addr_tokens = addr_norm.split()
        if name_tokens and addr_tokens:
            keys.append(f"cross_first:{name_tokens[0]}_{addr_tokens[0]}")
        
        return keys


class InvertedIndexBlocker:
    """Inverted index-based blocking for high recall candidate generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocking_config = config.get('blocking', {})
        self.key_generator = BlockingKeyGenerator(config)
        self.index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # key -> [(source, record_id)]
        self.source1_records: Dict[str, Dict] = {}
        self.source23_records: Dict[str, Dict] = {}  # source -> {id: record}
    
    def build_index(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> None:
        """Build inverted index from all sources."""
        logger.info("Building inverted index...")
        
        # Index source 1 records
        for _, row in source1.iterrows():
            record_id = row['id']
            self.source1_records[record_id] = {
                'id': record_id,
                'name': row['name'],
                'address': row['address']
            }
            keys = self.key_generator.generate_keys(row['name'], row['address'])
            for key in keys:
                self.index[key].append(('source1', record_id))
        
        # Index source 2 records
        for _, row in source2.iterrows():
            record_id = row['id']
            self.source23_records[('source2', record_id)] = {
                'id': record_id,
                'name': row['name'],
                'address': row['address'],
                'source': 'source2'
            }
            keys = self.key_generator.generate_keys(row['name'], row['address'])
            for key in keys:
                self.index[key].append(('source2', record_id))
        
        # Index source 3 records
        for _, row in source3.iterrows():
            record_id = row['id']
            self.source23_records[('source3', record_id)] = {
                'id': record_id,
                'name': row['name'],
                'address': row['address'],
                'source': 'source3'
            }
            keys = self.key_generator.generate_keys(row['name'], row['address'])
            for key in keys:
                self.index[key].append(('source3', record_id))
        
        logger.info(f"Built index with {len(self.index)} unique keys")
        logger.info(f"Source 1 records: {len(self.source1_records)}")
        logger.info(f"Source 2+3 records: {len(self.source23_records)}")
    
    def get_candidates(self, source1_id: str, max_candidates: int = 200) -> List[Tuple[str, str]]:
        """Get candidate matches for a source1 record."""
        if source1_id not in self.source1_records:
            return []
        
        record = self.source1_records[source1_id]
        keys = self.key_generator.generate_keys(record['name'], record['address'])
        
        # Collect candidates from all keys
        candidates: Dict[Tuple[str, str], int] = defaultdict(int)  # (source, id) -> key_count
        
        for key in keys:
            for source, cand_id in self.index.get(key, []):
                if source != 'source1':
                    candidates[(source, cand_id)] += 1
        
        # Sort by number of shared keys (descending)
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        
        # Limit candidates
        max_cand = self.blocking_config.get('max_candidates_per_source1', max_candidates)
        min_cand = self.blocking_config.get('min_candidates_per_source1', 1)
        
        result = [(source, cand_id) for (source, cand_id), _ in sorted_candidates[:max_cand]]
        
        # Ensure minimum candidates by adding from largest buckets if needed
        if len(result) < min_cand:
            # Fallback: add from most frequent keys
            all_candidates = []
            for key in keys:
                for source, cand_id in self.index.get(key, []):
                    if source != 'source1' and (source, cand_id) not in [(s, c) for s, c in result]:
                        all_candidates.append((source, cand_id))
            result.extend(all_candidates[:min_cand - len(result)])
        
        return result
    
    def generate_all_candidates(self, max_candidates: int = 200) -> List[Tuple[str, str, str]]:
        """Generate candidate pairs for all source1 records."""
        all_pairs = []
        
        for source1_id in self.source1_records:
            candidates = self.get_candidates(source1_id, max_candidates)
            for source, cand_id in candidates:
                all_pairs.append((source1_id, cand_id, source))
        
        logger.info(f"Generated {len(all_pairs)} candidate pairs")
        return all_pairs


class EmbeddingBlocker:
    """Embedding-based approximate nearest neighbor blocking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocking_config = config.get('blocking', {})
        self.model_name = self.blocking_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = self.blocking_config.get('embedding_dim', 384)
        self.ann_num_neighbors = self.blocking_config.get('ann_num_neighbors', 50)
        self.model = None
        self.index = None
        self.source23_ids = []
        self.source23_sources = []
    
    def _load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, skipping embedding blocking")
                return False
        return True
    
    def build_index(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> bool:
        """Build ANN index for source2+3 records."""
        if not self._load_model():
            return False
        
        logger.info("Building embedding-based ANN index...")
        
        # Prepare texts for source2+3
        texts = []
        self.source23_ids = []
        self.source23_sources = []
        
        for _, row in source2.iterrows():
            text = f"{row['name']} [SEP] {row['address']}"
            texts.append(text)
            self.source23_ids.append(row['id'])
            self.source23_sources.append('source2')
        
        for _, row in source3.iterrows():
            text = f"{row['name']} [SEP] {row['address']}"
            texts.append(text)
            self.source23_ids.append(row['id'])
            self.source23_sources.append('source3')
        
        if not texts:
            return False
        
        # Generate embeddings
        embeddings = self.model.encode(texts, batch_size=256, show_progress_bar=True, 
                                       convert_to_numpy=True, normalize_embeddings=True)
        
        # Build ANN index
        try:
            import hnswlib
            self.index = hnswlib.Index(space='ip', dim=self.embedding_dim)
            self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
            self.index.add_items(embeddings)
            self.index.set_ef(50)
            logger.info(f"Built HNSW index with {len(embeddings)} vectors")
        except ImportError:
            # Fallback to sklearn NearestNeighbors
            logger.info("hnswlib not available, using sklearn NearestNeighbors")
            self.index = NearestNeighbors(n_neighbors=min(self.ann_num_neighbors, len(embeddings)), 
                                          metric='cosine', algorithm='brute')
            self.index.fit(embeddings)
        
        return True
    
    def get_candidates(self, source1_record: Dict, k: int = None) -> List[Tuple[str, str, float]]:
        """Get nearest neighbors for a source1 record."""
        if self.index is None or self.model is None:
            return []
        
        k = k or self.ann_num_neighbors
        text = f"{source1_record['name']} [SEP] {source1_record['address']}"
        query_emb = self.model.encode([text], normalize_embeddings=True)
        
        if hasattr(self.index, 'knn_query'):
            # hnswlib
            labels, distances = self.index.knn_query(query_emb, k=min(k, len(self.source23_ids)))
            labels = labels[0]
            distances = distances[0]
        else:
            # sklearn
            distances, labels = self.index.kneighbors(query_emb, n_neighbors=min(k, len(self.source23_ids)))
            labels = labels[0]
            distances = distances[0]
        
        # Convert to (source, id, similarity)
        candidates = []
        for idx, dist in zip(labels, distances):
            if idx < len(self.source23_ids):
                sim = 1.0 - dist  # cosine similarity
                candidates.append((
                    self.source23_sources[idx],
                    self.source23_ids[idx],
                    float(sim)
                ))
        
        return candidates


class HybridBlocker:
    """Combines multiple blocking strategies for maximum recall."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blocking_config = config.get('blocking', {})
        self.inverted_blocker = InvertedIndexBlocker(config)
        self.embedding_blocker = EmbeddingBlocker(config) if self.blocking_config.get('use_embedding_blocking', True) else None
        self.use_embedding = self.blocking_config.get('use_embedding_blocking', True) and self.embedding_blocker is not None
    
    def build(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> None:
        """Build all blocking indices."""
        self.inverted_blocker.build_index(source1, source2, source3)
        
        if self.use_embedding:
            success = self.embedding_blocker.build_index(source1, source2, source3)
            if not success:
                self.use_embedding = False
                logger.warning("Embedding blocking disabled due to missing dependencies")
    
    def generate_candidates(self, max_candidates: int = 200) -> List[Tuple[str, str, str]]:
        """Generate candidates using hybrid approach."""
        all_pairs = []
        max_cand = self.blocking_config.get('max_candidates_per_source1', max_candidates)
        
        # Allocate budget between strategies
        inverted_budget = int(max_cand * 0.7)
        embedding_budget = max_cand - inverted_budget
        
        for source1_id, record in self.inverted_blocker.source1_records.items():
            candidates_set = set()
            
            # Inverted index candidates
            inv_candidates = self.inverted_blocker.get_candidates(source1_id, inverted_budget)
            for source, cand_id in inv_candidates:
                candidates_set.add((source, cand_id))
            
            # Embedding-based candidates
            if self.use_embedding:
                emb_candidates = self.embedding_blocker.get_candidates(record, embedding_budget)
                for source, cand_id, score in emb_candidates:
                    candidates_set.add((source, cand_id))
            
            # Convert to list
            for source, cand_id in list(candidates_set)[:max_cand]:
                all_pairs.append((source1_id, cand_id, source))
        
        logger.info(f"Hybrid blocking generated {len(all_pairs)} candidate pairs")
        return all_pairs
    
    def save_index(self, path: str) -> None:
        """Save blocking indices to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save inverted index
        with open(Path(path) / 'inverted_index.pkl', 'wb') as f:
            pickle.dump({
                'index': dict(self.inverted_blocker.index),
                'source1_records': self.inverted_blocker.source1_records,
                'source23_records': self.inverted_blocker.source23_records
            }, f)
        
        # Save embedding index if available
        if self.use_embedding and self.embedding_blocker.index is not None:
            import hnswlib
            if hasattr(self.embedding_blocker.index, 'save_index'):
                self.embedding_blocker.index.save_index(str(Path(path) / 'hnsw_index.bin'))
            else:
                with open(Path(path) / 'sklearn_index.pkl', 'wb') as f:
                    pickle.dump(self.embedding_blocker.index, f)
            
            with open(Path(path) / 'embedding_metadata.pkl', 'wb') as f:
                pickle.dump({
                    'source23_ids': self.embedding_blocker.source23_ids,
                    'source23_sources': self.embedding_blocker.source23_sources
                }, f)
        
        logger.info(f"Saved blocking indices to {path}")
    
    def load_index(self, path: str) -> None:
        """Load blocking indices from disk."""
        # Load inverted index
        with open(Path(path) / 'inverted_index.pkl', 'rb') as f:
            data = pickle.load(f)
            self.inverted_blocker.index = defaultdict(list, data['index'])
            self.inverted_blocker.source1_records = data['source1_records']
            self.inverted_blocker.source23_records = data['source23_records']
        
        # Load embedding index if available
        if self.use_embedding:
            try:
                import hnswlib
                hnsw_path = Path(path) / 'hnsw_index.bin'
                if hnsw_path.exists():
                    self.embedding_blocker.index = hnswlib.Index(space='ip', dim=self.embedding_blocker.embedding_dim)
                    self.embedding_blocker.index.load_index(str(hnsw_path))
                else:
                    with open(Path(path) / 'sklearn_index.pkl', 'rb') as f:
                        self.embedding_blocker.index = pickle.load(f)
                
                with open(Path(path) / 'embedding_metadata.pkl', 'rb') as f:
                    meta = pickle.load(f)
                    self.embedding_blocker.source23_ids = meta['source23_ids']
                    self.embedding_blocker.source23_sources = meta['source23_sources']
            except Exception as e:
                logger.warning(f"Could not load embedding index: {e}")
                self.use_embedding = False
        
        logger.info(f"Loaded blocking indices from {path}")


def run_blocking(config: Dict[str, Any], source1: pd.DataFrame, source2: pd.DataFrame, 
                 source3: pd.DataFrame, output_path: str) -> List[Tuple[str, str, str]]:
    """Main blocking pipeline function."""
    blocker = HybridBlocker(config)
    blocker.build(source1, source2, source3)
    candidates = blocker.generate_candidates()
    
    # Save candidate pairs
    from src.utils.io import write_candidate_pairs
    write_candidate_pairs(candidates, output_path)
    
    # Save index for reproducibility
    index_path = Path(output_path).parent / 'blocking_index'
    blocker.save_index(str(index_path))
    
    return candidates