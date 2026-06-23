"""Temporal prediction of detection difficulty using sliding windows."""

import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# PyTorch imports for LSTM
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LSTM model will not work.")


class DifficultyPredictor:
    """Predict delta_tp (strong - weak) from temporal runtime features.

    Two-stage model:
    1. Gatekeeper classifier: predicts if gain > 0
    2. Regressor: predicts gain magnitude (only trained on positive samples)
    """

    def __init__(self, window_size: int = 5, predict_horizon: int = 1, model_type: str = 'linear',
                 use_gatekeeper: bool = True, gatekeeper_type: str = 'rf',
                 gatekeeper_threshold: float = 0.3):
        """
        Initialize predictor.

        Args:
            window_size: Number of past frames to use as features (N)
            predict_horizon: Number of future frames to predict (M)
            model_type: 'linear', 'rf' (random forest), or 'xgboost'
            use_gatekeeper: Whether to use two-stage model with gatekeeper classifier
            gatekeeper_type: Classifier type for gatekeeper: 'rf' or 'xgboost'
            gatekeeper_threshold: Probability threshold for positive class (lower = higher recall)
                                  Default 0.3 to favor recall over precision
        """
        self.window_size = window_size
        self.predict_horizon = predict_horizon
        self.model_type = model_type
        self.target_column = 'delta_tp'
        self.use_gatekeeper = use_gatekeeper
        self.gatekeeper_type = gatekeeper_type
        self.gatekeeper_threshold = gatekeeper_threshold
        self.gatekeeper_stats = {}

        # Initialize gatekeeper classifier
        if use_gatekeeper:
            if gatekeeper_type == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    raise ImportError("XGBoost not available for gatekeeper")
                self.gatekeeper = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    base_score=0.5,
                    random_state=42,
                    n_jobs=-1,
                )
            else:  # default to RF
                self.gatekeeper = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    max_features='sqrt',
                    n_jobs=-1,
                )

        # Initialize regressor
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                max_features='sqrt',
                n_jobs=-1,
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = LinearRegression()
        
        self.feature_columns = [
            # Per-box: single lowest-confidence detection (bot0_conf == conf_min, so skip it)
            'bot0_area_log', 'bot0_aspect_ratio', 'bot0_track_changed',

            # Global features
            'det_count', 'delta_det_count',
            'conf_min', 'conf_max', 'conf_median', 'conf_entropy',
            'conf_top_k_gap_k2', 'conf_top_k_gap_k3', 'conf_top_k_gap_k5',
            'small_box_ratio', 'ratio_outlier_count', 'deformed_box_count',
            'intra_iou_mean', 'intra_iou_max',

            # Tracker features
            'track_continuity_ratio',
            'centroid_jitter_mean', 'centroid_jitter_max',
            'new_track_ratio', 'lost_track_count',
            'center_vanishing_ratio', 'high_conf_vanishing_ratio',
            'new_high_conf_ratio',

            # Motion-prediction features
            'ghost_prediction_error',
            'track_age_imbalance',
            'occlusion_risk_count',
        ]

    
    def load_video_data(self, csv_path: Path) -> pd.DataFrame:
        """Load cached features for a single video."""
        return pd.read_csv(csv_path)
    
    def create_temporal_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window features and targets from video DataFrame.

        Args:
            df: DataFrame with per-frame features

        Returns:
            (X, y) arrays for training/prediction
            X shape: (n_samples, window_size * n_features)
            y shape: (n_samples,) - target is mean of next M frames
        """
        X_list = []
        y_list = []

        # Ensure sorted by frame_id
        df = df.sort_values('frame_id').reset_index(drop=True)

        # Extract feature matrix
        features = df[self.feature_columns].values
        targets = df[self.target_column].values  # Shape: (n_frames,)

        # Create sliding windows
        for i in range(len(df) - self.window_size - self.predict_horizon + 1):
            # Past N frames as features
            window_features = features[i:i+self.window_size].flatten()

            # Future M frames' mean
            future_targets = targets[i+self.window_size:i+self.window_size+self.predict_horizon]
            target = np.mean(future_targets)

            X_list.append(window_features)
            y_list.append(target)

        return np.array(X_list), np.array(y_list)
    
    def prepare_dataset(self, data_dir: Path, video_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training dataset from multiple videos.
        
        Args:
            data_dir: Directory containing cached CSV files
            video_names: List of video names to include
            
        Returns:
            (X, y) combined from all videos
        """
        X_all = []
        y_all = []
        
        for video_name in video_names:
            csv_file = data_dir / f'{video_name}.csv'
            if not csv_file.exists():
                continue
            
            df = self.load_video_data(csv_file)
            X, y = self.create_temporal_features(df)
            
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
        
        if len(X_all) == 0:
            raise ValueError("No valid data found")
        
        return np.vstack(X_all), np.concatenate(y_all)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """Train the prediction model (two-stage if gatekeeper enabled)."""
        train_start_time = time.time()

        if self.use_gatekeeper:
            # Stage 1: Train gatekeeper classifier (gain > 0 vs gain <= 0)
            y_binary = (y_train > 0).astype(int)
            n_positive = y_binary.sum()
            n_negative = len(y_binary) - n_positive

            if verbose:
                print(f"  [Gatekeeper] Training classifier ({self.gatekeeper_type})...")
                print(f"    Positive samples (gain > 0): {n_positive} ({100*n_positive/len(y_binary):.1f}%)")
                print(f"    Negative samples (gain <= 0): {n_negative} ({100*n_negative/len(y_binary):.1f}%)")

            self.gatekeeper.fit(X_train, y_binary)

            # Store gatekeeper training stats
            self.gatekeeper_stats['n_positive_train'] = int(n_positive)
            self.gatekeeper_stats['n_negative_train'] = int(n_negative)
            self.gatekeeper_stats['positive_ratio_train'] = float(n_positive / len(y_binary))
            self.gatekeeper_stats['threshold'] = self.gatekeeper_threshold

            # Evaluate gatekeeper on training data (using custom threshold)
            y_binary_pred = self._gatekeeper_predict(X_train)
            self.gatekeeper_stats['train_accuracy'] = float(accuracy_score(y_binary, y_binary_pred))
            self.gatekeeper_stats['train_precision'] = float(precision_score(y_binary, y_binary_pred, zero_division=0))
            self.gatekeeper_stats['train_recall'] = float(recall_score(y_binary, y_binary_pred, zero_division=0))
            self.gatekeeper_stats['train_f1'] = float(f1_score(y_binary, y_binary_pred, zero_division=0))

            if verbose:
                print(f"    Threshold: {self.gatekeeper_threshold}")
                print(f"    Train Accuracy: {self.gatekeeper_stats['train_accuracy']:.4f}")
                print(f"    Train Precision: {self.gatekeeper_stats['train_precision']:.4f}")
                print(f"    Train Recall: {self.gatekeeper_stats['train_recall']:.4f}")
                print(f"    Train F1: {self.gatekeeper_stats['train_f1']:.4f}")

            # Stage 2: Train regressor only on positive samples
            positive_mask = y_train > 0
            X_positive = X_train[positive_mask]
            y_positive = y_train[positive_mask]

            if verbose:
                print(f"  [Regressor] Training on {len(X_positive)} positive samples...")

            if len(X_positive) > 0:
                self.model.fit(X_positive, y_positive)
            else:
                if verbose:
                    print("    Warning: No positive samples, regressor not trained")
        else:
            # Single-stage: train regressor on all data
            self.model.fit(X_train, y_train)

        self.train_time = time.time() - train_start_time
        if verbose:
            print(f"  Training completed in {self.train_time:.2f} seconds")

    def _gatekeeper_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using gatekeeper with custom threshold.

        Returns binary predictions (0 or 1) using probability threshold.
        Lower threshold = higher recall (more samples predicted as positive).
        """
        proba = self.gatekeeper.predict_proba(X)[:, 1]  # P(gain > 0)
        return (proba >= self.gatekeeper_threshold).astype(int)

    def predict(self, X: np.ndarray, record_time: bool = False) -> np.ndarray:
        """Predict difficulty change scores (delta_mAP).

        If gatekeeper is enabled:
        - Samples predicted as gain <= 0 get prediction = 0
        - Samples predicted as gain > 0 get regressor prediction
        """
        if record_time:
            inference_start_time = time.time()

        if self.use_gatekeeper:
            # Two-stage prediction with custom threshold
            gate_pred = self._gatekeeper_predict(X)  # 0 or 1
            predictions = np.zeros(len(X))

            # Only run regressor on samples predicted as positive
            positive_mask = gate_pred == 1
            if positive_mask.sum() > 0:
                predictions[positive_mask] = self.model.predict(X[positive_mask])
        else:
            predictions = self.model.predict(X)

        if record_time:
            self.inference_time = time.time() - inference_start_time
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get gatekeeper probability scores (only if gatekeeper is enabled)."""
        if not self.use_gatekeeper:
            raise ValueError("Gatekeeper not enabled")
        return self.gatekeeper.predict_proba(X)[:, 1]  # P(gain > 0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate prediction performance.

        Returns:
            Dictionary with MAE, RMSE, R2 scores for delta_tp
            If gatekeeper enabled, also includes gatekeeper metrics
        """
        y_pred = self.predict(X_test, record_time=True)

        result = {
            'predictions': y_pred,
            'ground_truth': y_test,
            'train_time': self.train_time,
            'inference_time': self.inference_time,
            'num_samples': len(X_test),
            'target_column': self.target_column,
        }

        # Overall regression metrics
        result['mae'] = mean_absolute_error(y_test, y_pred)
        result['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        result['r2'] = r2_score(y_test, y_pred)
        rho, pval = spearmanr(y_test, y_pred)
        result['spearman'] = rho
        result['spearman_p'] = pval
 
        # Gatekeeper-specific evaluation
        if self.use_gatekeeper:
            y_binary_true = (y_test > 0).astype(int)
            y_binary_pred = self._gatekeeper_predict(X_test)

            result['gatekeeper_accuracy'] = float(accuracy_score(y_binary_true, y_binary_pred))
            result['gatekeeper_precision'] = float(precision_score(y_binary_true, y_binary_pred, zero_division=0))
            result['gatekeeper_recall'] = float(recall_score(y_binary_true, y_binary_pred, zero_division=0))
            result['gatekeeper_f1'] = float(f1_score(y_binary_true, y_binary_pred, zero_division=0))

            # Stats on test set
            n_positive = y_binary_true.sum()
            result['n_positive_test'] = int(n_positive)
            result['n_negative_test'] = int(len(y_binary_true) - n_positive)
            result['positive_ratio_test'] = float(n_positive / len(y_binary_true))

            # Regression metrics only on true positive samples (where gain > 0)
            true_positive_mask = y_test > 0
            if true_positive_mask.sum() > 0:
                y_test_pos = y_test[true_positive_mask]
                y_pred_pos = y_pred[true_positive_mask]
                result['mae_positive_only'] = mean_absolute_error(y_test_pos, y_pred_pos)
                result['rmse_positive_only'] = np.sqrt(mean_squared_error(y_test_pos, y_pred_pos))
                if len(y_test_pos) > 1:
                    result['r2_positive_only'] = r2_score(y_test_pos, y_pred_pos)
                    rho_pos, pval_pos = spearmanr(y_test_pos, y_pred_pos)
                    result['spearman_positive_only'] = rho_pos
                    result['spearman_p_positive_only'] = pval_pos

            # Copy gatekeeper training stats
            result['gatekeeper_stats'] = self.gatekeeper_stats.copy()

        return result
    
    def save(self, path: Path):
        """Save trained model to disk."""
        save_data = {
            'model': self.model,
            'window_size': self.window_size,
            'predict_horizon': self.predict_horizon,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'train_time': self.train_time,
            'use_gatekeeper': self.use_gatekeeper,
        }
        if self.use_gatekeeper:
            save_data['gatekeeper'] = self.gatekeeper
            save_data['gatekeeper_type'] = self.gatekeeper_type
            save_data['gatekeeper_threshold'] = self.gatekeeper_threshold
            save_data['gatekeeper_stats'] = self.gatekeeper_stats

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    @classmethod
    def load(cls, path: Path):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        predictor = cls(
            window_size=data['window_size'],
            predict_horizon=data['predict_horizon'],
            model_type=data['model_type'],
            use_gatekeeper=data.get('use_gatekeeper', False),
            gatekeeper_type=data.get('gatekeeper_type', 'rf'),
            gatekeeper_threshold=data.get('gatekeeper_threshold', 0.3),
        )
        predictor.model = data['model']
        predictor.feature_columns = data['feature_columns']
        predictor.target_column = data.get('target_column', 'delta_tp')
        predictor.train_time = data.get('train_time', None)

        if predictor.use_gatekeeper and 'gatekeeper' in data:
            predictor.gatekeeper = data['gatekeeper']
            predictor.gatekeeper_stats = data.get('gatekeeper_stats', {})

        return predictor
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (for Random Forest and XGBoost)."""
        if self.model_type not in ('rf', 'xgboost'):
            return None

        # Create feature names (feature_name_t-N, feature_name_t-N+1, ...)
        feature_names = []
        for t in range(self.window_size):
            for feat in self.feature_columns:
                feature_names.append(f'{feat}_t-{self.window_size-t}')

        importances = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return df
