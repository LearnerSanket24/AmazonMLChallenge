#!/usr/bin/env python3
"""Validation script for Amazon ML Challenge 2026 submission format."""

import argparse
import pandas as pd
import sys
from pathlib import Path


def validate_submission(submission_path: str, ground_truth_path: str = None) -> bool:
    """Validate submission file format and optionally compute score."""
    
    print(f"Validating: {submission_path}")
    
    try:
        # Read with explicit tab separator
        df = pd.read_csv(submission_path, sep='\t', dtype=str, keep_default_na=False, na_filter=False)
    except Exception as e:
        print(f"❌ ERROR: Failed to read file: {e}")
        return False
    
    # Check columns
    expected_cols = ['source1_id', 'matched_ids']
    if list(df.columns) != expected_cols:
        print(f"❌ ERROR: Invalid columns. Expected {expected_cols}, got {list(df.columns)}")
        return False
    print(f"✓ Columns: {list(df.columns)}")
    
    # Check for duplicates
    dup_count = df['source1_id'].duplicated().sum()
    if dup_count > 0:
        print(f"❌ ERROR: {dup_count} duplicate source1_id(s) found")
        return False
    print(f"✓ No duplicate source1_ids")
    
    # Check matched_ids format
    empty_count = 0
    non_empty_count = 0
    total_matched = 0
    
    for idx, row in df.iterrows():
        matched = row['matched_ids']
        if matched == '' or matched is None:
            empty_count += 1
        else:
            non_empty_count += 1
            ids = matched.split(',')
            # Check for empty IDs in list
            if any(not id.strip() for id in ids):
                print(f"❌ ERROR: Empty ID in matched_ids at row {idx}: '{matched}'")
                return False
            total_matched += len(ids)
    
    print(f"✓ Matched_ids format valid")
    print(f"  - Entities with matches: {non_empty_count}")
    print(f"  - Singleton predictions (empty): {empty_count}")
    print(f"  - Total predicted matches: {total_matched}")
    print(f"  - Total entities: {len(df)}")
    
    # If ground truth provided, compute macro F0.5
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"\nComputing score against: {ground_truth_path}")
        gt = pd.read_csv(ground_truth_path, sep='\t', dtype=str, keep_default_na=False, na_filter=False)
        
        if list(gt.columns) != expected_cols:
            print(f"⚠ WARNING: Ground truth columns mismatch: {list(gt.columns)}")
        
        # Build lookup dicts
        gt_dict = {}
        for _, row in gt.iterrows():
            s1_id = row['source1_id']
            matched = row['matched_ids']
            if matched and matched != '':
                gt_dict[s1_id] = set(matched.split(','))
            else:
                gt_dict[s1_id] = set()
        
        pred_dict = {}
        for _, row in df.iterrows():
            s1_id = row['source1_id']
            matched = row['matched_ids']
            if matched and matched != '':
                pred_dict[s1_id] = set(matched.split(','))
            else:
                pred_dict[s1_id] = set()
        
        # Compute macro F0.5
        f05_scores = []
        all_gt_ids = set(gt_dict.keys())
        all_pred_ids = set(pred_dict.keys())
        
        # Check coverage
        missing_in_pred = all_gt_ids - all_pred_ids
        extra_in_pred = all_pred_ids - all_gt_ids
        
        if missing_in_pred:
            print(f"⚠ WARNING: {len(missing_in_pred)} source1_ids in ground truth but not in prediction")
        if extra_in_pred:
            print(f"⚠ WARNING: {len(extra_in_pred)} source1_ids in prediction but not in ground truth")
        
        for s1_id in all_gt_ids:
            true_matches = gt_dict[s1_id]
            pred_matches = pred_dict.get(s1_id, set())
            
            if not true_matches and not pred_matches:
                # True singleton, correctly predicted
                f05_scores.append(1.0)
            elif not true_matches and pred_matches:
                # True singleton, but predicted matches (FP)
                f05_scores.append(0.0)
            else:
                # Has true matches
                tp = len(true_matches & pred_matches)
                fp = len(pred_matches - true_matches)
                fn = len(true_matches - pred_matches)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if precision + recall == 0:
                    f05 = 0.0
                else:
                    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                
                f05_scores.append(f05)
        
        macro_f05 = sum(f05_scores) / len(f05_scores) if f05_scores else 0.0
        print(f"\n📊 Macro F0.5 Score: {macro_f05:.6f}")
        
        # Per-entity breakdown
        correct_singletons = sum(1 for s in f05_scores if s == 1.0)
        zero_scores = sum(1 for s in f05_scores if s == 0.0)
        partial_scores = len(f05_scores) - correct_singletons - zero_scores
        
        print(f"  - Perfect (1.0): {correct_singletons}")
        print(f"  - Zero (0.0): {zero_scores}")
        print(f"  - Partial (0-1): {partial_scores}")
    
    print(f"\n✅ Validation PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate submission format for Amazon ML Challenge 2026')
    parser.add_argument('submission', help='Path to matching_results.tsv')
    parser.add_argument('--ground-truth', '-g', help='Path to ground_truth_train.tsv for scoring')
    
    args = parser.parse_args()
    
    success = validate_submission(args.submission, args.ground_truth)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()