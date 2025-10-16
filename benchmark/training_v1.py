"""
Enhanced Machine Learning Pipeline with Early Stopping and Monitoring
Author: Enhanced ML Pipeline
Date: 2025
"""

import os
import time
import joblib
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# ML Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, log_loss
)
from sklearn.preprocessing import label_binarize
from memory_profiler import memory_usage

# Configuration
CONFIG = {
    'BATCH_SIZE': 1000,
    'MEMORY_INTERVAL': 0.1,
    'TOP_K_CLASSES': 100,
    'BASE_DIR': '/Users/tirtasetiawan/Documents/rki_v1/rki_2025/model',  # Changed from '/kaggle/working/svd'
    'ENABLE_MEMORY_PROFILING': True,
    'MAX_EPOCHS': 100,
    'PATIENCE': 10,
    'MIN_DELTA': 0.001,
    'VALIDATION_SPLIT': 0.2
}

class EnhancedDataLoader:
    """
    Enhanced Data Loader untuk menggabungkan data dari hasil reduksi dan vektorisasi
    
    Fitur:
    - Load data hasil reduksi dengan filter train_features/test_features
    - Load data hasil vektorisasi dengan filter y_train/y_test
    - Filter berdasarkan level taxonomi dan kmer length
    - Auto-discovery dataset yang available
    - Validasi kelengkapan data
    """
    
    def __init__(self, reduction_log_path, vectorization_paths_file):
        """
        Initialize EnhancedDataLoader
        
        Args:
            reduction_log_path: Path ke output_paths_log.txt dari hasil reduksi
            vectorization_paths_file: Path ke all_vectorization_output_paths.txt
        """
        self.reduction_log_path = reduction_log_path
        self.vectorization_paths_file = vectorization_paths_file
        self.reduction_data = self._load_reduction_log()
        self.vectorization_paths = self._load_vectorization_paths()
        
    def _load_reduction_log(self):
        """Load dan parse reduction log file"""
        try:
            reduction_data = []
            with open(self.reduction_log_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        # Handle case where there might be more than 6 parts
                        timestamp = parts[0]
                        level = parts[1]
                        kmer = parts[2]
                        method = parts[3]
                        device = parts[4]
                        file_type = parts[5]
                        file_path = parts[6] if len(parts) > 6 else ""
                        
                        reduction_data.append({
                            'timestamp': timestamp,
                            'level': level,
                            'kmer': kmer,
                            'method': method,
                            'device': device,
                            'file_type': file_type,
                            'file_path': file_path
                        })
            return reduction_data
        except Exception as e:
            print(f"❌ Error loading reduction log: {e}")
            return []
    
    def _load_vectorization_paths(self):
        """Load vectorization paths file"""
        try:
            with open(self.vectorization_paths_file, 'r') as f:
                paths = [line.strip() for line in f if line.strip()]
            return paths
        except Exception as e:
            print(f"❌ Error loading vectorization paths: {e}")
            return []
    
    def filter_reduction_data(self, level=None, kmer=None, file_types=['features_train', 'features_test']):
        """
        Filter reduction data berdasarkan kriteria
        
        Args:
            level: Level taxonomi (class, family, genus, etc.)
            kmer: Kmer length (k6, k8, k10)
            file_types: List of file types to include (default: train/test features)
        """
        filtered = self.reduction_data.copy()
        
        if level:
            filtered = [item for item in filtered if item['level'] == level]
        
        if kmer:
            filtered = [item for item in filtered if item['kmer'] == kmer]
            
        if file_types:
            filtered = [item for item in filtered if item['file_type'] in file_types]
            
        return filtered
    
    def filter_vectorization_data(self, level=None, kmer=None, prefixes=['y_train', 'y_test']):
        """
        Filter vectorization paths berdasarkan kriteria
        
        Args:
            level: Level taxonomi (class, family, genus, etc.)
            kmer: Kmer length (k6, k8, k10)
            prefixes: List of filename prefixes to include
        """
        filtered_paths = []
        
        for path in self.vectorization_paths:
            # Extract level and kmer from path
            path_parts = path.split('/')
            
            # Check if path contains level and kmer
            level_match = True if not level else level in path_parts
            kmer_match = True if not kmer else any(kmer in part for part in path_parts)
            
            # Check prefix
            filename = os.path.basename(path)
            prefix_match = any(filename.startswith(prefix) for prefix in prefixes)
            
            if level_match and kmer_match and prefix_match:
                # Extract level and kmer info from path
                extracted_level = None
                extracted_kmer = None
                
                for part in path_parts:
                    if part in ['class', 'family', 'genus', 'kingdom', 'order', 'phylum', 'species']:
                        extracted_level = part
                    if part.startswith('k') and part[1:].isdigit():
                        extracted_kmer = part
                
                filtered_paths.append({
                    'path': path,
                    'level': extracted_level,
                    'kmer': extracted_kmer,
                    'filename': filename,
                    'file_type': filename.split('_')[0] + '_' + filename.split('_')[1]  # y_train, y_test
                })
        
        return filtered_paths
    
    def discover_available_datasets(self):
        """
        Discover semua kombinasi level-kmer yang tersedia
        """
        # Get unique combinations from reduction data
        reduction_combos = set()
        for item in self.reduction_data:
            if item['file_type'] in ['features_train', 'features_test']:
                reduction_combos.add((item['level'], item['kmer']))
        
        # Get unique combinations from vectorization data
        vectorization_combos = set()
        vectorization_filtered = self.filter_vectorization_data()
        for item in vectorization_filtered:
            if item['level'] and item['kmer']:
                vectorization_combos.add((item['level'], item['kmer']))
        
        # Find intersection (datasets yang memiliki both reduction and vectorization data)
        available_datasets = reduction_combos.intersection(vectorization_combos)
        
        return sorted(list(available_datasets))
    
    def load_dataset_for_pipeline(self, level, kmer, strategy='smart_aggregate'):
        """
        Load complete dataset untuk ML pipeline
        
        Args:
            level: Level taxonomi (class, family, genus, etc.)
            kmer: Kmer length (k6, k8, k10)
            strategy: 'smart_aggregate' (single method, all folds), 
                     'aggregate_reduction' (all files), or 
                     'single_fold' (first file only)
            
        Returns:
            dict: {
                'X_train_reduction': features dari reduksi (train),
                'X_test_reduction': features dari reduksi (test), 
                'y_train': labels dari vektorisasi,
                'y_test': labels dari vektorisasi,
                'metadata': informasi tambahan
            }
        """
        result = {
            'X_train_reduction': None,
            'X_test_reduction': None,
            'y_train': None,
            'y_test': None,
            'metadata': {
                'level': level,
                'kmer': kmer,
                'strategy': strategy,
                'reduction_files': [],
                'vectorization_files': []
            }
        }
        
        try:
            # Strategy: Smart Aggregate (single method, all folds)
            if strategy == 'smart_aggregate':
                reduction_filtered = self.filter_reduction_data(
                    level=level, 
                    kmer=kmer, 
                    file_types=['features_train', 'features_test']
                )
                
                # Group by method
                from collections import defaultdict
                method_groups = defaultdict(list)
                
                for item in reduction_filtered:
                    file_path = item['file_path']
                    if 'umap' in file_path.lower():
                        method_groups['umap'].append(item)
                    elif 'svd' in file_path.lower():
                        method_groups['svd'].append(item)
                    elif 'pca' in file_path.lower():
                        method_groups['pca'].append(item)
                    else:
                        method_groups['other'].append(item)
                
                # Choose method with most files (most complete)
                best_method = max(method_groups.keys(), key=lambda k: len(method_groups[k])) if method_groups else None
                
                if best_method and method_groups[best_method]:
                    print(f"📊 Using method: {best_method} ({len(method_groups[best_method])} files)")
                    
                    train_dfs = []
                    test_dfs = []
                    
                    for item in method_groups[best_method]:
                        file_path = item['file_path']
                        if not os.path.exists(file_path):
                            continue
                            
                        try:
                            df = pd.read_csv(file_path, comment='#')
                            
                            if item['file_type'] == 'features_train':
                                train_dfs.append(df)
                                print(f"✅ Train: {df.shape} from {os.path.basename(file_path)}")
                                result['metadata']['reduction_files'].append(file_path)
                                
                            elif item['file_type'] == 'features_test':
                                test_dfs.append(df)
                                print(f"✅ Test: {df.shape} from {os.path.basename(file_path)}")
                                result['metadata']['reduction_files'].append(file_path)
                                
                        except Exception as e:
                            print(f"⚠️ Error loading {file_path}: {e}")
                            continue
                    
                    # Combine DataFrames dengan consistent columns
                    if train_dfs:
                        # Find common columns across all train DataFrames
                        common_cols = set(train_dfs[0].columns)
                        for df in train_dfs[1:]:
                            common_cols = common_cols.intersection(set(df.columns))
                        
                        # Only use common columns untuk consistency
                        common_cols = sorted([col for col in common_cols if col.startswith('component_')])
                        
                        print(f"🔄 Using {len(common_cols)} common features: {common_cols}")
                        
                        aligned_dfs = []
                        for df in train_dfs:
                            aligned_dfs.append(df[common_cols + ['label', 'method', 'device', 'fold']])
                        
                        result['X_train_reduction'] = pd.concat(aligned_dfs, ignore_index=True)
                        print(f"🔄 Combined train_features: {result['X_train_reduction'].shape}")
                    
                    if test_dfs:
                        # Use same common columns for test data
                        if 'common_cols' in locals() and common_cols:
                            aligned_test_dfs = []
                            for df in test_dfs:
                                # Only use common component columns available in test
                                available_cols = [col for col in common_cols if col in df.columns]
                                other_cols = [col for col in ['label', 'method', 'device', 'fold'] if col in df.columns]
                                aligned_test_dfs.append(df[available_cols + other_cols])
                            
                            result['X_test_reduction'] = pd.concat(aligned_test_dfs, ignore_index=True)
                            print(f"🔄 Combined test_features: {result['X_test_reduction'].shape}")
                        else:
                            result['X_test_reduction'] = pd.concat(test_dfs, ignore_index=True)
                            print(f"🔄 Combined test_features: {result['X_test_reduction'].shape}")
                        
            elif strategy == 'aggregate_reduction':
                # Strategy 1: Aggregate all reduction data from different folds
                all_train_dfs = []
                all_test_dfs = []
                
                reduction_filtered = self.filter_reduction_data(
                    level=level, 
                    kmer=kmer, 
                    file_types=['features_train', 'features_test']
                )
                
                for item in reduction_filtered:
                    file_path = item['file_path']
                    if not os.path.exists(file_path):
                        continue
                        
                    try:
                        df = pd.read_csv(file_path, comment='#')
                        
                        if item['file_type'] == 'features_train':
                            all_train_dfs.append(df)
                            print(f"✅ Added train_features: {df.shape} from {os.path.basename(file_path)}")
                            result['metadata']['reduction_files'].append(file_path)
                            
                        elif item['file_type'] == 'features_test':
                            all_test_dfs.append(df)
                            print(f"✅ Added test_features: {df.shape} from {os.path.basename(file_path)}")
                            result['metadata']['reduction_files'].append(file_path)
                            
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
                        continue
                
                # Combine all DataFrames
                if all_train_dfs:
                    result['X_train_reduction'] = pd.concat(all_train_dfs, ignore_index=True)
                    print(f"🔄 Combined train_features: {result['X_train_reduction'].shape}")
                
                if all_test_dfs:
                    result['X_test_reduction'] = pd.concat(all_test_dfs, ignore_index=True)
                    print(f"🔄 Combined test_features: {result['X_test_reduction'].shape}")
                    
            else:
                # Strategy 3: Single fold approach (original implementation)
                train_loaded = False
                test_loaded = False
                
                reduction_filtered = self.filter_reduction_data(
                    level=level, 
                    kmer=kmer, 
                    file_types=['features_train', 'features_test']
                )
                
                for item in reduction_filtered:
                    file_path = item['file_path']
                    if not os.path.exists(file_path):
                        print(f"⚠️ File not found: {file_path}")
                        continue
                        
                    try:
                        if item['file_type'] == 'features_train' and not train_loaded:
                            result['X_train_reduction'] = pd.read_csv(file_path, comment='#')
                            result['metadata']['reduction_files'].append(file_path)
                            print(f"✅ Loaded train_features: {result['X_train_reduction'].shape}")
                            train_loaded = True
                            
                        elif item['file_type'] == 'features_test' and not test_loaded:
                            result['X_test_reduction'] = pd.read_csv(file_path, comment='#')
                            result['metadata']['reduction_files'].append(file_path)
                            print(f"✅ Loaded test_features: {result['X_test_reduction'].shape}")
                            test_loaded = True
                            
                    except Exception as csv_error:
                        print(f"⚠️ Error reading CSV {file_path}: {csv_error}")
                        continue
            
            # 2. Load vectorization labels (y_train dan y_test)
            vectorization_filtered = self.filter_vectorization_data(
                level=level,
                kmer=kmer,
                prefixes=['y_train', 'y_test']
            )
            
            for item in vectorization_filtered:
                if item['file_type'] == 'y_train' and item['filename'].endswith('.npy'):
                    result['y_train'] = np.load(item['path'])
                    result['metadata']['vectorization_files'].append(item['path'])
                elif item['file_type'] == 'y_test' and item['filename'].endswith('.npy'):
                    result['y_test'] = np.load(item['path'])
                    result['metadata']['vectorization_files'].append(item['path'])
            
            # 3. Validasi kelengkapan data
            missing_components = []
            if result['X_train_reduction'] is None:
                missing_components.append('X_train_reduction')
            if result['X_test_reduction'] is None:
                missing_components.append('X_test_reduction')
            if result['y_train'] is None:
                missing_components.append('y_train')
            if result['y_test'] is None:
                missing_components.append('y_test')
            
            if missing_components:
                print(f"⚠️ Missing components for {level} {kmer}: {missing_components}")
                return None
            
            # 4. Validasi dimensi data
            if len(result['X_train_reduction']) != len(result['y_train']):
                print(f"⚠️ Dimension mismatch for {level} {kmer}: X_train({len(result['X_train_reduction'])}) vs y_train({len(result['y_train'])})")
                return None
                
            if len(result['X_test_reduction']) != len(result['y_test']):
                print(f"⚠️ Dimension mismatch for {level} {kmer}: X_test({len(result['X_test_reduction'])}) vs y_test({len(result['y_test'])})")
                return None
            
            # 5. Align features between train and test
            if result['X_train_reduction'] is not None and result['X_test_reduction'] is not None:
                # Get component columns
                train_component_cols = [col for col in result['X_train_reduction'].columns if col.startswith('component_')]
                test_component_cols = [col for col in result['X_test_reduction'].columns if col.startswith('component_')]
                
                if train_component_cols and test_component_cols:
                    # Use minimum number of features for consistency
                    min_features = min(len(train_component_cols), len(test_component_cols))
                    
                    # Select first N component columns from both
                    selected_train_cols = train_component_cols[:min_features]
                    selected_test_cols = test_component_cols[:min_features]
                    
                    print(f"🔄 Feature alignment: train {len(train_component_cols)} → {min_features}, test {len(test_component_cols)} → {min_features}")
                    
                    # Extract aligned features
                    result['X_train'] = result['X_train_reduction'][selected_train_cols].fillna(0)
                    result['X_test'] = result['X_test_reduction'][selected_test_cols].fillna(0)
                    
                    print(f"🎯 Aligned features - X_train: {result['X_train'].shape}, X_test: {result['X_test'].shape}")
                else:
                    print("⚠️ No component columns found in reduction data")
            
            print(f"✅ Successfully loaded dataset: {level} {kmer}")
            print(f"   - X_train_reduction: {result['X_train_reduction'].shape}")
            print(f"   - X_test_reduction: {result['X_test_reduction'].shape}")
            print(f"   - X_train (aligned): {result.get('X_train', 'None')}")
            print(f"   - X_test (aligned): {result.get('X_test', 'None')}")
            print(f"   - y_train: {result['y_train'].shape}")
            print(f"   - y_test: {result['y_test'].shape}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading dataset {level} {kmer}: {e}")
            return None
    
    def batch_load_all_datasets(self, levels=None, kmers=None):
        """
        Load semua dataset yang tersedia dalam batch
        
        Args:
            levels: List level yang ingin di-load (None = semua)
            kmers: List kmer yang ingin di-load (None = semua)
            
        Returns:
            dict: {(level, kmer): dataset_dict}
        """
        available_datasets = self.discover_available_datasets()
        loaded_datasets = {}
        
        for level, kmer in available_datasets:
            # Filter berdasarkan input parameters
            if levels and level not in levels:
                continue
            if kmers and kmer not in kmers:
                continue
                
            print(f"\n🔄 Loading dataset: {level} {kmer}")
            dataset = self.load_dataset_for_pipeline(level, kmer)
            
            if dataset:
                loaded_datasets[(level, kmer)] = dataset
            else:
                print(f"❌ Failed to load: {level} {kmer}")
        
        print(f"\n📊 Summary: Loaded {len(loaded_datasets)} out of {len(available_datasets)} available datasets")
        return loaded_datasets
    
    def get_dataset_summary(self):
        """
        Generate summary informasi dataset yang tersedia
        """
        available = self.discover_available_datasets()
        
        summary = {
            'total_datasets': len(available),
            'levels': list(set([combo[0] for combo in available])),
            'kmers': list(set([combo[1] for combo in available])),
            'combinations': available
        }
        
        print("📋 **DATASET SUMMARY**")
        print(f"Total Available Datasets: {summary['total_datasets']}")
        print(f"Levels: {summary['levels']}")
        print(f"Kmers: {summary['kmers']}")
        print("\n📁 Available Combinations:")
        for level, kmer in available:
            print(f"   - {level} × {kmer}")
            
        return summary

class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving."""

    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'min':
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

class TrainingMonitor:
    """Monitor training progress with metrics and plotting."""

    def __init__(self, model_name, level, method):
        self.model_name = model_name
        self.level = level
        self.method = method
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'val_roc_auc': []
        }

    def update(self, epoch, metrics):
        """Update training history."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - {self.level} - {self.method} Training History', fontsize=16)

        # Loss plot
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        #axes[0, 0].grid(True)

        # Accuracy plot
        axes[0, 1].plot(self.history['epoch'], self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(self.history['epoch'], self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        #axes[0, 1].grid(True)

        # F1 Score plot
        axes[1, 0].plot(self.history['epoch'], self.history['train_f1'], 'b-', label='Train F1')
        axes[1, 0].plot(self.history['epoch'], self.history['val_f1'], 'r-', label='Val F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        #axes[1, 0].grid(True)

        # ROC AUC plot
        axes[1, 1].plot(self.history['epoch'], self.history['val_roc_auc'], 'g-', label='Val ROC AUC')
        axes[1, 1].set_title('ROC AUC')
        axes[1, 1].set_xlabel('False Positive Rate') # Corrected label
        axes[1, 1].set_ylabel('True Positive Rate') # Corrected label
        axes[1, 1].legend()
        #axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    → Training plot saved to: {save_path}")

        plt.show()

    def save_history(self, save_path):
        """Save training history to CSV."""
        df = pd.DataFrame(self.history)
        df.to_csv(save_path, index=False)
        print(f"    → Training history saved to: {save_path}")

class EnhancedMLPipeline:
    """Enhanced ML Pipeline with robust training and monitoring."""

    def __init__(self, config=None):
        self.config = config or CONFIG

    def ensure_numpy_arrays(self, X_train, X_test, y_train, y_test):
        """Convert DataFrames to numpy arrays to avoid indexing issues."""
        print("    → Converting data to numpy arrays if needed...")

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            print(f"    → Converted X_train DataFrame to numpy array: {X_train.shape}")

        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
            print(f"    → Converted X_test DataFrame to numpy array: {X_test.shape}")

        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            print(f"    → Converted y_train Series to numpy array: {y_train.shape}")

        if isinstance(y_test, pd.Series):
            y_test = y_test.values
            print(f"    → Converted y_test Series to numpy array: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    def run_and_profile_memory(self, func, *args, **kwargs):
        """Profile memory usage during function execution."""
        if not self.config['ENABLE_MEMORY_PROFILING']:
            result = func(*args, **kwargs)
            return 0, 0, result

        mem_usage, result = memory_usage(
            (func, args, kwargs),
            max_usage=False,
            retval=True,
            interval=self.config['MEMORY_INTERVAL'],
            include_children=True
        )
        return np.mean(mem_usage), np.max(mem_usage), result

    def split_train_validation(self, X_train, y_train, val_split=0.2):
        """Split training data into train and validation sets."""
        # Ensure we're working with numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        n_samples = X_train.shape[0]
        n_val = int(n_samples * val_split)

        # Random shuffle
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]

        return X_train_split, y_train_split, X_val, y_val

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')

        # Loss (if probabilities available)
        if y_prob is not None:
            try:
                metrics['loss'] = log_loss(y_true, y_prob)
            except:
                metrics['loss'] = np.nan

            # ROC AUC for multiclass
            try:
                unique_labels = np.unique(y_true)
                if len(unique_labels) > 2:
                    y_true_bin = label_binarize(y_true, classes=unique_labels)
                    if y_prob.shape[1] == len(unique_labels):
                        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                    else:
                        metrics['roc_auc'] = np.nan
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except Exception as e:
                print(f"    → [ERROR] ROC-AUC calculation error: {e}")
                metrics['roc_auc'] = np.nan
        else:
            metrics['loss'] = np.nan
            metrics['roc_auc'] = np.nan

        return metrics

    def fit_sgd_with_monitoring(self, model, X_train, y_train, level, method):
        """Fit SGD model with early stopping and monitoring."""
        print(f"    → Training SGD with monitoring and early stopping...")

        # Split data for validation
        X_train_split, y_train_split, X_val, y_val = self.split_train_validation(
            X_train, y_train, self.config['VALIDATION_SPLIT']
        )

        # Setup class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))

        # Initialize monitoring
        monitor = TrainingMonitor(model.__class__.__name__, level, method)
        early_stopping = EarlyStopping(
            patience=self.config['PATIENCE'],
            min_delta=self.config['MIN_DELTA'],
            mode='min'  # Monitor validation loss
        )

        # Training loop
        batch_size = self.config['BATCH_SIZE']
        best_model_state = None
        best_val_loss = float('inf')

        for epoch in range(self.config['MAX_EPOCHS']):
            print(f"    → Epoch {epoch + 1}/{self.config['MAX_EPOCHS']}")

            # Shuffle training data
            indices = np.random.permutation(X_train_split.shape[0])
            X_train_shuffled = X_train_split[indices]
            y_train_shuffled = y_train_split[indices]

            # Batch training
            epoch_train_losses = []

            for start in tqdm(range(0, X_train_split.shape[0], batch_size), desc="Training batches"):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Sample weights for batch
                sample_weight = np.array([class_weight_dict[label] for label in y_batch])

                # Partial fit
                if epoch == 0 and start == 0:
                    model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
                else:
                    model.partial_fit(X_batch, y_batch, sample_weight=sample_weight)

                # Calculate batch loss (approximate)
                try:
                    # Ensure X_batch is dense for predict_proba if model doesn't support sparse
                    if hasattr(model, 'predict_proba'):
                         if hasattr(X_batch, 'toarray'):
                             y_prob_batch = model.predict_proba(X_batch.toarray())
                         else:
                             y_prob_batch = model.predict_proba(X_batch)
                         batch_loss = log_loss(y_batch, y_prob_batch)
                         epoch_train_losses.append(batch_loss)
                except Exception as e:
                    # print(f"    → [DEBUG] Batch loss calculation failed: {e}")
                    pass


            # Evaluate on training set (sample for efficiency)
            sample_size = min(5000, X_train_split.shape[0])
            train_indices = np.random.choice(X_train_split.shape[0], sample_size, replace=False)
            X_train_sample = X_train_split[train_indices]
            y_train_sample = y_train_split[train_indices]

            # Ensure X_train_sample is dense if needed
            if hasattr(model, 'predict') and hasattr(X_train_sample, 'toarray'):
                X_train_sample_dense = X_train_sample.toarray()
            else:
                X_train_sample_dense = X_train_sample # Use as is if already dense or model handles sparse

            y_train_pred = model.predict(X_train_sample_dense)
            y_train_prob = None
            if hasattr(model, 'predict_proba'):
                y_train_prob = model.predict_proba(X_train_sample_dense)

            train_metrics = self.calculate_metrics(y_train_sample, y_train_pred, y_train_prob)

            # Evaluate on validation set
            # Ensure X_val is dense if needed
            if hasattr(model, 'predict') and hasattr(X_val, 'toarray'):
                X_val_dense = X_val.toarray()
            else:
                X_val_dense = X_val # Use as is if already dense or model handles sparse

            y_val_pred = model.predict(X_val_dense)
            y_val_prob = None
            if hasattr(model, 'predict_proba'):
                y_val_prob = model.predict_proba(X_val_dense)
            val_metrics = self.calculate_metrics(y_val, y_val_pred, y_val_prob)

            # Update monitoring
            metrics_dict = {
                'train_loss': train_metrics.get('loss', np.nan),
                'val_loss': val_metrics.get('loss', np.nan),
                'train_acc': train_metrics.get('accuracy', np.nan),
                'val_acc': val_metrics.get('accuracy', np.nan),
                'train_f1': train_metrics.get('f1', np.nan),
                'val_f1': val_metrics.get('f1', np.nan),
                'val_roc_auc': val_metrics.get('roc_auc', np.nan)
            }


            monitor.update(epoch + 1, metrics_dict)

            # Print progress
            train_loss_str = f"{metrics_dict['train_loss']:.4f}" if not np.isnan(metrics_dict['train_loss']) else 'N/A'
            val_loss_str = f"{metrics_dict['val_loss']:.4f}" if not np.isnan(metrics_dict['val_loss']) else 'N/A'
            print(f"        Train Loss: {train_loss_str}, Val Loss: {val_loss_str}")
            print(f"        Train Acc: {metrics_dict['train_acc']:.4f}, Val Acc: {metrics_dict['val_acc']:.4f}")
            print(f"        Train F1: {metrics_dict['train_f1']:.4f}, Val F1: {metrics_dict['val_f1']:.4f}")


            # Save best model based on validation loss
            if not np.isnan(val_metrics.get('loss', np.nan)):
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    # Store model state (this is model-specific, SGDClassifier has coef_ and intercept_)
                    # Need a more general way for other models or just save the model file
                    try:
                        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                              best_model_state = {
                                  'coef_': model.coef_.copy(),
                                  'intercept_': model.intercept_.copy(),
                                  'classes_': model.classes_.copy() if hasattr(model, 'classes_') else None
                              }
                        # Add handling for other model types if needed
                        # elif isinstance(model, SomeOtherModel):
                        #    best_model_state = model.get_state() # Assume a method exists
                        else:
                             best_model_state = None # Cannot save state generically
                    except Exception as e:
                        print(f"    → [DEBUG] Failed to capture model state: {e}")
                        best_model_state = None


            # Early stopping check
            if not np.isnan(val_metrics.get('loss', np.nan)):
                if early_stopping(val_metrics['loss']):
                    print(f"    → Early stopping at epoch {epoch + 1}")
                    break
            elif early_stopping(metrics_dict.get('val_acc', 0)): # Fallback to accuracy if loss is NaN
                 print(f"    → Early stopping at epoch {epoch + 1} (based on validation accuracy)")
                 break


            # Memory cleanup
            if epoch % 10 == 0:
                gc.collect()

        # Restore best model state if available
        if best_model_state and hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
             try:
                model.coef_ = best_model_state['coef_']
                model.intercept_ = best_model_state['intercept_']
                if best_model_state['classes_'] is not None and hasattr(model, 'classes_'):
                     model.classes_ = best_model_state['classes_']
                print(f"    → Restored best model state (Val Loss: {best_val_loss:.4f})")
             except Exception as e:
                print(f"    → [DEBUG] Failed to restore model state: {e}")


        # Save training history and plots
        history_path = f"{level}_{method}_{model.__class__.__name__}_training_history.csv"
        plot_path = f"{level}_{method}_{model.__class__.__name__}_training_plot.png"

        monitor.save_history(history_path)
        monitor.plot_training_history(plot_path)

        return model

    def fit_model_safely(self, model, X_train, y_train, level, method, batch_size=1000):
        """Enhanced model fitting with different strategies for different models."""
        # Ensure X_train is dense if needed by the model
        if hasattr(model, 'fit') or hasattr(model, 'partial_fit'):
            if hasattr(X_train, 'toarray'):
                X_train_dense = X_train.toarray()
            else:
                X_train_dense = X_train # Use as is if already dense


        if hasattr(model, 'partial_fit') and isinstance(model, SGDClassifier):
            return self.fit_sgd_with_monitoring(model, X_train_dense, y_train, level, method) # Pass dense X_train
        elif hasattr(model, 'partial_fit'):
            print("    → Using standard partial_fit with tqdm...")
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))

            for start in tqdm(range(0, X_train_dense.shape[0], batch_size), desc="Training batches"):
                end = start + batch_size
                X_batch = X_train_dense[start:end] # Use dense batch
                y_batch = y_train[start:end]
                sample_weight = np.array([class_weight_dict[label] for label in y_batch])

                if start == 0:
                    model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
                else:
                    model.partial_fit(X_batch, y_batch, sample_weight=sample_weight)

            return model
        else:
            print("    → Using standard fit()...")
            model.fit(X_train_dense, y_train) # Use dense X_train
            return model


    def debug_confidence_distribution(self, y_prob, model_name):
        """Debug confidence distribution to understand model behavior."""
        confidences = np.max(y_prob, axis=1)

        print(f"\n=== {model_name} Confidence Analysis ===")
        print(f"Confidence shape: {confidences.shape}")
        print(f"Confidence range: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
        print(f"Mean confidence: {np.mean(confidences):.4f}")
        print(f"Std confidence: {np.std(confidences):.4f}")
        print(f"Samples with confidence = 1.0: {np.sum(confidences == 1.0)}")
        print(f"Samples with confidence > 0.99: {np.sum(confidences > 0.99)}")

        # Plot confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{model_name} - Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        #plt.grid(True, alpha=0.3)
        plt.savefig(f"{model_name}_confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()

        return confidences

    def calculate_robust_confidence(self, y_prob, method="max"):
        """Calculate confidence with various methods."""
        if y_prob is None:
            # Ensure y_pred is defined or accessible here if y_prob is None
            # This might require passing y_pred explicitly or handling the case where y_prob is None earlier
            # For now, returning array of NaNs with length matching y_true or X_test size
            # Assuming y_test is available in the scope where this function is called or passed as argument
            # A safer approach is to calculate confidence only when y_prob is not None.
            return np.full(len(y_test), np.nan) # Use len(y_test) for length if y_prob is None

        if method == "max":
            return np.max(y_prob, axis=1)
        elif method == "entropy":
            # Avoid log(0)
            y_prob = np.clip(y_prob, 1e-10, 1)
            entropy = -np.sum(y_prob * np.log(y_prob), axis=1)
            max_entropy = np.log(y_prob.shape[1])
            # Avoid division by zero if only one class
            if max_entropy == 0:
                 return np.ones(y_prob.shape[0])
            return 1 - (entropy / max_entropy)
        elif method == "margin":
             if y_prob.shape[1] < 2:
                  return np.zeros(y_prob.shape[0]) # Margin is 0 if only one class
             sorted_probs = np.sort(y_prob, axis=1)
             return sorted_probs[:, -1] - sorted_probs[:, -2]
        elif method == "ratio":
             if y_prob.shape[1] < 2:
                 return np.ones(y_prob.shape[0]) # Ratio is 1 if only one class
             sorted_probs = np.sort(y_prob, axis=1)
             # Avoid division by zero
             denominator = sorted_probs[:, -2]
             denominator[denominator == 0] = 1e-10
             return sorted_probs[:, -1] / denominator
        else:
             return np.full(y_prob.shape[0], np.nan) # Unknown method


    def predict_safely(self, model, X):
        """Perform prediction and probability prediction safely."""
        # Ensure X is dense if needed by the model
        if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
             if hasattr(X, 'toarray'):
                 X_dense = X.toarray()
             else:
                 X_dense = X # Use as is if already dense

        y_pred = model.predict(X_dense)

        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_dense)
            except Exception as e:
                print(f"    → [WARNING] Could not get prediction probabilities: {e}")

        return y_pred, y_prob

    def evaluate_roc_auc(self, y_true, y_prob, name="", level="", method="", top_k=None, average="macro"):
        """Enhanced ROC AUC evaluation with better error handling."""
        # Import required modules at function level to avoid scope issues
        from sklearn.metrics import roc_auc_score, roc_curve
        import matplotlib.pyplot as plt
        import numpy as np
        
        if y_prob is None:
             print(f"   -> ROC AUC {name} {level} {method}: N/A (No probabilities)")
             return np.nan

        # Ensure y_prob is a numpy array and is 2D
        y_prob = np.asarray(y_prob)
        if y_prob.ndim != 2:
            print(f"    → [ERROR] y_prob has unexpected dimensions: {y_prob.ndim}. Expected 2.")
            # Attempt to reshape if it's 1D, otherwise return nan
            if y_prob.ndim == 1:
                 print("    → Attempting to reshape 1D y_prob to 2D.")
                 y_prob = y_prob.reshape(-1, 1)
            else:
                 return np.nan

        # Use robust fallback calculation approach
        try:
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            # Binary classification case
            if n_classes == 2 and y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                print(f"   -> ROC AUC {name} {level} {method}: {roc_auc:.4f} (binary)")
                
                # Plot ROC curve
                plt.figure(figsize=(7, 5))
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {name}')
                plt.legend(loc='lower right')
                plt.grid(True, alpha=0.3)
                plt.show()
                return roc_auc
            
            # Binary classification with single probability column
            elif n_classes == 2 and y_prob.shape[1] == 1:
                roc_auc = roc_auc_score(y_true, y_prob[:, 0])
                print(f"   -> ROC AUC {name} {level} {method}: {roc_auc:.4f} (binary single col)")
                return roc_auc
            
            # Multiclass case - use one-vs-rest approach
            elif n_classes > 2 and y_prob.shape[1] >= n_classes:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                print(f"   -> ROC AUC {name} {level} {method}: {roc_auc:.4f} (multiclass ovr)")
                return roc_auc
            
            # Single class case
            elif n_classes == 1:
                print(f"   -> ROC AUC {name} {level} {method}: N/A (Single class)")
                return np.nan
            
            # Shape mismatch case
            else:
                print(f"   -> ROC AUC {name} {level} {method}: N/A (Shape mismatch: {n_classes} classes, {y_prob.shape[1]} prob cols)")
                return np.nan
                
        except Exception as e:
            print(f"❌ ROC AUC calculation failed: {e}")
            return np.nan



    def save_predictions_safely(self, df_pred, filename, model_name, is_first_model=False):
        """Save predictions with robust error handling."""
        try:
            # Check if file exists and is not empty before appending
            if not is_first_model and os.path.exists(filename) and os.path.getsize(filename) > 0:
                 df_pred.to_csv(filename, index=False, mode='a', header=False)
                 print(f"✅ Appended predictions to: {filename}")
            else:
                 df_pred.to_csv(filename, index=False, mode='w')
                 print(f"✅ Created new prediction file (or overwrote empty one): {filename}")


            # Validate file content
            try:
                 test_df = pd.read_csv(filename)
                 print(f"📊 File now has {len(test_df)} rows and columns: {test_df.columns.tolist()}")
            except Exception as e:
                 print(f"❌ Error reading back saved file {filename}: {e}")


        except Exception as e:
            print(f"❌ Error saving predictions for {model_name}: {e}")
            fallback_filename = filename.replace('.csv', f'_{model_name}_error.csv') # Added error suffix
            df_pred.to_csv(fallback_filename, index=False)
            print(f"💾 Saved to fallback file: {fallback_filename}")


def create_enhanced_models():
    """Create model configurations with enhanced SGD."""
    return {
        # "SGD_Enhanced": SGDClassifier(
        #     loss='log_loss',
        #     max_iter=1,  # Will be controlled by our custom training loop
        #     tol=None,
        #     random_state=42,
        #     learning_rate='adaptive',
        #     warm_start=True,
        #     eta0=0.01,
        #     verbose=0,
        #     alpha=0.0001,  # L2 regularization
        #     l1_ratio=0.15,  # ElasticNet mixing parameter
        #     penalty='elasticnet'  # ElasticNet regularization
        # ),
        "KNN_Bayes": KNeighborsClassifier(n_neighbors=5),
        "GaussianNaiveBayes": GaussianNB(),
        # "LogisticRegression": LogisticRegression(max_iter=1000),
        # "SVC": SVC(probability=True),
        # "RandomForest": RandomForestClassifier(n_estimators=100,
        #                                        criterion= "gini",
        #                                        max_depth= None,
        #                                        min_samples_split= 2,
        #                                        min_samples_leaf= 1,
        #                                        min_weight_fraction_leaf= 0.0),
        # "XGBoost": XGBClassifier(n_estimators=100,
        #                           learning_rate=0.1,
        #                           max_depth=3),
        # "LightGBM": LGBMClassifier(n_estimators=100,
        #                              learning_rate=0.1,
        #                              max_depth=3),
    }

# Main execution function (to be called from main script)
def run_enhanced_ml_pipeline(X_train, X_test, y_train, y_test, label_encoder, level, method):
    """Run the enhanced ML pipeline."""

    # Initialize pipeline
    pipeline = EnhancedMLPipeline(CONFIG)

    # Ensure all data is in numpy array format to avoid indexing issues
    print("= DATA PREPARATION =")
    X_train, X_test, y_train, y_test = pipeline.ensure_numpy_arrays(X_train, X_test, y_train, y_test)

    # Create models
    models = create_enhanced_models()

    results = []
    filename_predictions = f"{level}_{method}_enhanced_all_predictions.csv"
    filename_eval = f"{level}_{method}_enhanced_model_evaluation.csv"


    # Clear previous prediction file if it exists for a fresh start
    if os.path.exists(filename_predictions):
        try:
            os.remove(filename_predictions)
            print(f"Cleared existing prediction file: {filename_predictions}")
        except Exception as e:
             print(f"Warning: Could not clear existing prediction file {filename_predictions}: {e}. Appending instead.")


    print(f"= ENHANCED MACHINE LEARNING PIPELINE {level} =")
    print(f"Config: {CONFIG}")
    print(f"Data shapes (after conversion): X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Label shapes: y_train={y_train.shape}, y_test={y_test.shape}")
    print(f"Unique classes: {len(np.unique(y_train))}")


    for i, (name, model) in enumerate(tqdm(models.items(), desc="Training models", total=len(models))):

        dir_path = CONFIG['BASE_DIR']
        os.makedirs(dir_path, exist_ok=True) # Ensure base directory exists
        mod_path = os.path.join(dir_path, f"{level}_{method}_{name}_enhanced_model.pkl")

        # Training phase
        if not os.path.exists(mod_path):
            print(f"\n== Training {name} {level} {method} ==")

            start_time = time.time()
            # Pass numpy arrays directly
            avg_mem_train, peak_mem_train, model = pipeline.run_and_profile_memory(
                pipeline.fit_model_safely, model, X_train, y_train, level, method
            )
            end_time = time.time()

            train_time = end_time - start_time

            print(f"    → Save Enhanced Model to pkl {name} {level} {method}")
            try:
                joblib.dump(model, mod_path)
                print(f"✅ Model saved to: {mod_path}")
            except Exception as e:
                print(f"❌ Error saving model {name} to {mod_path}: {e}")
        else:
            print(f"== Enhanced model {mod_path} already exists. Loading...")
            try:
                model = joblib.load(mod_path)
                print(f"✅ Model loaded from: {mod_path}")
            except Exception as e:
                 print(f"❌ Error loading model {mod_path}: {e}. Skipping this model.")
                 continue # Skip to next model if loading fails

            train_time = 0  # Not measured for pre-trained models
            avg_mem_train, peak_mem_train = 0, 0 # Not measured for pre-trained models


        # Prediction phase
        print(f"    → Enhanced Prediction {name} {level}")

        start_pred_time = time.time()
        # Use direct prediction on the full test set (dense)
        avg_mem_pred, peak_mem_pred, (y_pred, y_prob) = pipeline.run_and_profile_memory(
             pipeline.predict_safely, model, X_test
        )
        end_pred_time = time.time()

        predict_time = end_pred_time - start_pred_time

        # Enhanced confidence analysis (only if probabilities are available)
        if y_prob is not None:
            print(f"    → Analyzing confidence distribution for {name}")
            confidences = pipeline.debug_confidence_distribution(y_prob, name)
        else:
             print(f"    → Skipping confidence analysis for {name} (No probabilities)")
             confidences = np.full(len(y_test), np.nan) # Placeholder


        # Transform labels - handle case where label_encoder is None
        try:
           if label_encoder is not None:
               pred_labels = label_encoder.inverse_transform(y_pred)
               true_labels = label_encoder.inverse_transform(y_test)
           else:
               # Use encoded labels directly if no label encoder is available
               pred_labels = y_pred
               true_labels = y_test
               print(f"    → Using encoded labels for {name} (no label encoder available)")
        except Exception as e:
           print(f"❌ Error transforming labels for {name}: {e}. Using encoded labels.")
           pred_labels = y_pred # Use encoded if transformation fails
           true_labels = y_test


        # Create enhanced prediction DataFrame
        print(f"    → Creating enhanced prediction DataFrame for {name}")
        df_pred = pd.DataFrame()
        df_pred["Model"] = f"{level}_{name}_enhanced"
        df_pred["True_Label"] = true_labels
        df_pred["Pred_Label"] = pred_labels
        df_pred["Confidence_Max"] = pipeline.calculate_robust_confidence(y_prob, "max")
        df_pred["Confidence_Entropy"] = pipeline.calculate_robust_confidence(y_prob, "entropy")
        df_pred["Confidence_Margin"] = pipeline.calculate_robust_confidence(y_prob, "margin")


        # Save predictions
        is_first_model = (i == 0)
        pipeline.save_predictions_safely(df_pred, filename_predictions, name, is_first_model)

        # Calculate metrics
        print(f"=== Enhanced Metrics Calculation {name} {level} {method} ===")

        # Ensure y_test and y_pred are compatible for metrics
        # If label transformation failed, metrics might be on encoded labels
        # If transformation succeeded, metrics are on original labels
        # Let's use the potentially untransformed y_test and y_pred if transformation failed
        y_test_for_metrics = y_test
        y_pred_for_metrics = y_pred


        acc = accuracy_score(y_test_for_metrics, y_pred_for_metrics)

        # F1, Recall, Precision require multi-class handling for macro average
        try:
            f1 = f1_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
        except Exception as e:
            print(f"❌ Error calculating F1: {e}")
            f1 = np.nan

        try:
            recall = recall_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
        except Exception as e:
             print(f"❌ Error calculating Recall: {e}")
             recall = np.nan

        try:
            precision = precision_score(y_test_for_metrics, y_pred_for_metrics, average='macro')
        except Exception as e:
             print(f"❌ Error calculating Precision: {e}")
             precision = np.nan


        print(f"   -> ACC {name}: {acc:.4f}")
        print(f"   -> F1 {name}: {f1:.4f}")
        print(f"   -> RECALL {name}: {recall:.4f}")
        print(f"   -> PRECISION {name}: {precision:.4f}")

        # Enhanced ROC AUC (calculate on encoded labels if transformation failed)
        roc_auc = pipeline.evaluate_roc_auc(
            y_true=y_test, # Always use original encoded y_test for ROC AUC
            y_prob=y_prob,
            name=name,
            level=level,
            method=method,
            top_k=CONFIG['TOP_K_CLASSES'],
            average="macro"
        )


        # Store results
        results.append({
            "Model": f"{level}_{name}_enhanced",
            "Accuracy": acc,
            "F1_Macro": f1,
            "Recall_Macro": recall,
            "Precision_Macro": precision,
            "ROC_AUC": roc_auc,
            "Train_Time_Seconds": round(train_time, 3),
            "Predict_Time_Seconds": round(predict_time, 3),
            "Peak_Memory_Train_MB": round(peak_mem_train, 2),
            "Avg_Memory_Train_MB": round(avg_mem_train, 2),
            "Peak_Memory_Predict_MB": round(peak_mem_pred, 2),
            "Avg_Memory_Predict_MB": round(avg_mem_pred, 2)
        })

        # Save incremental results
        print(f"= Save Enhanced Metrics to csv {name} {level}")
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename_eval, index=False)

        # Display current results
        metrics_eval = results_df.sort_values(by='F1_Macro', ascending=False)
        print(f"Current best F1: {metrics_eval.iloc[0]['F1_Macro']:.4f}")

        # Memory cleanup
        gc.collect()

    # Final results
    print(f"\n= FINAL ENHANCED RESULTS {level} {method} =")
    final_results = pd.DataFrame(results)
    # final_results.to_csv(f"{level}_{method}_final_enhanced_evaluation.csv", index=False) # Already saved incrementally

    # Check if results is empty
    if final_results.empty:
        print("❌ No models successfully processed.")
        return pd.DataFrame(), None

    best_model_row = final_results.sort_values(by="F1_Macro", ascending=False).iloc[0]
    best_model_name = best_model_row['Model']
    best_f1 = best_model_row['F1_Macro']


    print(f"✅ Best enhanced model for {level} {method}: {best_model_name}")
    print(f"✅ Best F1 Score: {best_f1:.4f}")

    return final_results, best_model_name

def debug_data_loading():
    """Debug function to check data loading and types"""
    print("="*50)
    print("DEBUGGING DATA LOADING")
    print("="*50)

    try:
        # Load data
        print("Loading X_train...")
        X_train = pd.read_csv(csv_path_X_train)
        print(f"X_train loaded: {type(X_train)}, shape: {X_train.shape}")
        print(f"X_train columns (first 5): {X_train.columns.tolist()[:5]}")
        print(f"X_train index: {type(X_train.index)}, first 5: {X_train.index[:5].tolist()}")

        print("\nLoading X_test...")
        X_test = pd.read_csv(csv_path_X_test)
        print(f"X_test loaded: {type(X_test)}, shape: {X_test.shape}")

        print("\nLoading y_train...")
        y_train = pd.read_csv('/content/y_train_svd.csv')['y_train'].values
        print(f"y_train loaded: {type(y_train)}, shape: {y_train.shape}")
        print(f"y_train unique values: {len(np.unique(y_train))}")
        print(f"y_train first 10 values: {y_train[:10]}")

        print("\nLoading y_test...")
        y_test = pd.read_csv('/content/y_test_svd.csv')['y_test'].values
        print(f"y_test loaded: {type(y_test)}, shape: {y_test.shape}")

        print("\nLoading label_encoder...")
        with open('/content/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"label_encoder loaded: {type(label_encoder)}")
        print(f"label_encoder classes: {len(label_encoder.classes_)} classes")
        print(f"First 10 classes: {label_encoder.classes_[:10]}")

        # Convert to numpy arrays
        print("\nConverting to numpy arrays...")
        X_train_np = X_train.values
        X_test_np = X_test.values

        print(f"X_train_np: {type(X_train_np)}, shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
        print(f"X_test_np: {type(X_test_np)}, shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")

        # Test indexing
        print("\nTesting indexing...")
        indices = np.random.permutation(min(100, X_train_np.shape[0]))[:min(20, X_train_np.shape[0])]
        print(f"Test indices: {indices}")

        try:
            X_sample = X_train_np[indices]
            print(f"Numpy indexing successful: {X_sample.shape}")
        except Exception as e:
            print(f"Numpy indexing failed: {e}")

        try:
            X_sample_df = X_train.iloc[indices]
            print(f"DataFrame iloc indexing successful: {X_sample_df.shape}")
        except Exception as e:
            print(f"DataFrame iloc indexing failed: {e}")

        return X_train_np, X_test_np, y_train, y_test, label_encoder

    except Exception as e:
        print(f"Error in data loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def debug_simple_train_val_split():
    """Debug simple train validation split"""
    print("="*50)
    print("DEBUGGING TRAIN-VAL SPLIT")
    print("="*50)

    X_train_np, X_test_np, y_train, y_test, label_encoder = debug_data_loading()

    if X_train_np is None:
        return

    try:
        # Simple split
        n_samples = X_train_np.shape[0]
        n_val = int(n_samples * 0.2)

        print(f"Total samples: {n_samples}")
        print(f"Validation samples: {n_val}")
        print(f"Training samples: {n_samples - n_val}")

        # Random indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        print(f"Train indices shape: {train_indices.shape}")
        print(f"Val indices shape: {val_indices.shape}")
        print(f"Train indices first 10: {train_indices[:10]}")
        print(f"Val indices first 10: {val_indices[:10]}")

        # Test splitting
        X_train_split = X_train_np[train_indices]
        y_train_split = y_train[train_indices]
        X_val = X_train_np[val_indices]
        y_val = y_val[val_indices]

        print(f"X_train_split shape: {X_train_split.shape}")
        print(f"y_train_split shape: {y_train_split.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")

        print("Train-val split successful!")
        return True

    except Exception as e:
        print(f"Error in train-val split: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_simple_model_training():
    """Debug simple model training"""
    print("="*50)
    print("DEBUGGING SIMPLE MODEL TRAINING")
    print("="*50)

    X_train_np, X_test_np, y_train, y_test, label_encoder = debug_data_loading()

    if X_train_np is None:
        return

    try:
        # Simple model training
        print("Training Gaussian Naive Bayes...")
        model = GaussianNB()

        # Use small subset for testing
        subset_size = min(10000, X_train_np.shape[0])
        indices = np.random.choice(X_train_np.shape[0], subset_size, replace=False)

        X_subset = X_train_np[indices]
        y_subset = y_train[indices]

        print(f"Training subset shape: {X_subset.shape}")
        print(f"Training subset labels shape: {y_subset.shape}")

        model.fit(X_subset, y_subset)
        print("Model training successful!")

        # Test prediction
        test_subset_size = min(1000, X_test_np.shape[0])
        test_indices = np.random.choice(X_test_np.shape[0], test_subset_size, replace=False)
        X_test_subset = X_test_np[test_indices]
        y_test_subset = y_test[test_indices]

        y_pred = model.predict(X_test_subset)
        y_prob = model.predict_proba(X_test_subset)

        print(f"Predictions shape: {y_pred.shape}")
        print(f"Probabilities shape: {y_prob.shape}")

        acc = accuracy_score(y_test_subset, y_pred)
        print(f"Test accuracy: {acc:.4f}")

        print("Simple model training successful!")
        return True

    except Exception as e:
        print(f"Error in simple model training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
# csv_path_X_train = '/content/X_train_svd_genus_353.csv'
# csv_path_X_test = '/content/X_test_svd_genus_353.csv'
# y_train = pd.read_csv('/content/y_train_svd.csv')['y_train'].values
# y_test = pd.read_csv('/content/y_test_svd.csv')['y_test'].values

# X_train = pd.read_csv(csv_path_X_train)
# X_test = pd.read_csv(csv_path_X_test)

# print(f"Train label count: {len(np.unique(y_train))}")
# print(f"Test label count: {len(np.unique(y_test))}")

# # X_train_sparse.shape[0] == len(y_train)
# # X_test_sparse.shape[0] == len(y_test)


# with open('/content/label_encoder.pkl', 'rb') as f:
#     label_encoder = pickle.load(f)

"""
Fixed Usage example for Enhanced ML Pipeline
"""

# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

# # Your existing imports and data loading
# method = "svd"
# level = 'genus'

# print(f"= STEP 5: Enhanced Machine Learning {level} =")

# # Load your data (replace with your actual data loading)
# print("Loading data...")
# try:
#     X_train = pd.read_csv(csv_path_X_train)
#     X_test = pd.read_csv(csv_path_X_test)

#     y_train = pd.read_csv('/content/y_train_svd.csv')['y_train'].values
#     y_test = pd.read_csv('/content/y_test_svd.csv')['y_test'].values
#     with open('/content/label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)

#     print(f"Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
#     print(f"Label shapes: y_train={y_train.shape}, y_test={y_test.shape}")
#     print(f"Unique classes: {len(np.unique(y_train))}")

#     print("Data loading completed!")

#     # Import the enhanced pipeline after successful data loading
#     print("Importing enhanced pipeline...")
#     #from enhanced_ml_pipeline import run_enhanced_ml_pipeline, CONFIG
#     print("Enhanced pipeline imported successfully!")

#     # Run the enhanced pipeline
#     print("Starting enhanced pipeline...")
#     results_df, best_model_name = run_enhanced_ml_pipeline(
#         X_train=X_train,     # Will be converted to numpy internally
#         X_test=X_test,       # Will be converted to numpy internally
#         y_train=y_train,
#         y_test=y_test,
#         label_encoder=label_encoder,
#         level=level,
#         method=method
#     )

#     print("\n" + "="*50)
#     print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
#     print("="*50)
#     print(f"Best Model: {best_model_name}")
#     print("\nFinal Results:")
#     print(results_df.sort_values('F1_Macro', ascending=False))

# except Exception as e:
#     print(f"Error in enhanced pipeline: {e}")
#     import traceback
#     traceback.print_exc()

    # Run debug version if main pipeline fails
    # print("\n" + "="*50)
    # print("RUNNING DEBUG VERSION")
    # print("="*50)

    # try:
    #     exec(open('/tmp/debug_enhanced_pipeline.py').read())
    # except:
    #     print("Debug version also failed")


def run_enhanced_pipeline_with_enhanced_loader(level, kmer, reduction_log_path, vectorization_paths_file):
    """
    Run enhanced ML pipeline menggunakan EnhancedDataLoader
    
    Args:
        level: Level taxonomi (class, family, genus, etc.)
        kmer: Kmer length (k6, k8, k10)
        reduction_log_path: Path ke output_paths_log.txt
        vectorization_paths_file: Path ke all_vectorization_output_paths.txt
        
    Returns:
        tuple: (results_df, best_model_name, dataset_info)
    """
    print(f"\n🚀 **ENHANCED PIPELINE with EnhancedDataLoader**")
    print(f"Target Dataset: {level} × {kmer}")
    print("="*60)
    
    try:
        # 1. Initialize EnhancedDataLoader
        print("📁 Initializing EnhancedDataLoader...")
        loader = EnhancedDataLoader(reduction_log_path, vectorization_paths_file)
        
        # 2. Load dataset
        print(f"📊 Loading dataset: {level} {kmer}")
        dataset = loader.load_dataset_for_pipeline(level, kmer)
        
        if dataset is None:
            print(f"❌ Failed to load dataset: {level} {kmer}")
            return None, None, None
        
        # 3. Prepare data for pipeline
        # Extract only numeric feature columns and handle NaN
        if dataset['X_train_reduction'] is not None:
            # Get only component columns (numeric features)
            feature_cols = [col for col in dataset['X_train_reduction'].columns 
                          if col.startswith('component_')]
            X_train_df = dataset['X_train_reduction'][feature_cols]
            
            # Handle NaN values
            if X_train_df.isnull().any().any():
                print("⚠️ Found NaN values in X_train, filling with 0...")
                X_train_df = X_train_df.fillna(0)
                
            X_train = X_train_df.values
        else:
            X_train = None
            
        if dataset['X_test_reduction'] is not None:
            # Get only component columns (numeric features)  
            feature_cols_test = [col for col in dataset['X_test_reduction'].columns 
                               if col.startswith('component_')]
            X_test_df = dataset['X_test_reduction'][feature_cols_test]
            
            # Handle NaN values
            if X_test_df.isnull().any().any():
                print("⚠️ Found NaN values in X_test, filling with 0...")
                X_test_df = X_test_df.fillna(0)
                
            X_test = X_test_df.values
        else:
            X_test = None
        y_train = dataset['y_train']
        y_test = dataset['y_test']
        
        print(f"✅ Data loaded successfully:")
        print(f"   - X_train (reduction): {X_train.shape}")
        print(f"   - X_test (reduction): {X_test.shape}")
        print(f"   - y_train (vectorization): {y_train.shape}")
        print(f"   - y_test (vectorization): {y_test.shape}")
        print(f"   - Unique classes: {len(np.unique(y_train))}")
        
        # 4. Run enhanced ML pipeline
        print("\n🤖 Starting Enhanced ML Pipeline...")
        results_df, best_model_name = run_enhanced_ml_pipeline(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            label_encoder=None,  # Will be handled internally
            level=level,
            method=f"reduction_{kmer}"
        )
        
        return results_df, best_model_name, dataset['metadata']
        
    except Exception as e:
        print(f"❌ Error in enhanced pipeline with loader: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def batch_process_with_enhanced_loader(reduction_log_path, vectorization_paths_file, 
                                     levels=None, kmers=None):
    """
    Batch processing semua dataset menggunakan EnhancedDataLoader
    
    Args:
        reduction_log_path: Path ke output_paths_log.txt
        vectorization_paths_file: Path ke all_vectorization_output_paths.txt
        levels: List level yang ingin di-process (None = semua)
        kmers: List kmer yang ingin di-process (None = semua)
        
    Returns:
        dict: Results dari semua dataset yang berhasil di-process
    """
    print("\n🔥 **BATCH PROCESSING with EnhancedDataLoader**")
    print("="*60)
    
    # Initialize loader
    loader = EnhancedDataLoader(reduction_log_path, vectorization_paths_file)
    
    # Get dataset summary
    summary = loader.get_dataset_summary()
    
    # Filter datasets berdasarkan input
    available_datasets = loader.discover_available_datasets()
    target_datasets = []
    
    for level, kmer in available_datasets:
        if levels and level not in levels:
            continue
        if kmers and kmer not in kmers:
            continue
        target_datasets.append((level, kmer))
    
    print(f"\n🎯 Target Datasets: {len(target_datasets)}")
    for level, kmer in target_datasets:
        print(f"   - {level} × {kmer}")
    
    # Batch processing
    all_results = {}
    successful_runs = 0
    failed_runs = 0
    
    for i, (level, kmer) in enumerate(target_datasets, 1):
        print(f"\n📊 Processing {i}/{len(target_datasets)}: {level} × {kmer}")
        print("-" * 40)
        
        start_time = time.time()
        results_df, best_model, metadata = run_enhanced_pipeline_with_enhanced_loader(
            level, kmer, reduction_log_path, vectorization_paths_file
        )
        end_time = time.time()
        
        if results_df is not None:
            all_results[(level, kmer)] = {
                'results_df': results_df,
                'best_model': best_model,
                'metadata': metadata,
                'processing_time': end_time - start_time
            }
            successful_runs += 1
            print(f"✅ Completed in {end_time - start_time:.2f}s")
        else:
            failed_runs += 1
            print(f"❌ Failed after {end_time - start_time:.2f}s")
    
    # Summary
    print("\n" + "="*60)
    print("🏁 **BATCH PROCESSING SUMMARY**")
    print("="*60)
    print(f"✅ Successful: {successful_runs}")
    print(f"❌ Failed: {failed_runs}")
    print(f"📊 Total: {len(target_datasets)}")
    
    return all_results


def get_enhanced_loader_example_usage():
    """
    Contoh penggunaan EnhancedDataLoader
    """
    example_code = '''
# ================================
# CONTOH PENGGUNAAN EnhancedDataLoader
# ================================

# 1. SETUP PATHS
reduction_log = "/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction/output_paths_log.txt"
vectorization_paths = "/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization_config/all_vectorization_output_paths.txt"

# 2. SINGLE DATASET PROCESSING
results_df, best_model, metadata = run_enhanced_pipeline_with_enhanced_loader(
    level="class",
    kmer="k10", 
    reduction_log_path=reduction_log,
    vectorization_paths_file=vectorization_paths
)

# 3. BATCH PROCESSING
batch_results = batch_process_with_enhanced_loader(
    reduction_log_path=reduction_log,
    vectorization_paths_file=vectorization_paths,
    levels=["class", "family", "genus"],  # Optional filter
    kmers=["k6", "k8", "k10"]            # Optional filter
)

# 4. DATASET DISCOVERY
loader = EnhancedDataLoader(reduction_log, vectorization_paths)
available_datasets = loader.discover_available_datasets()
summary = loader.get_dataset_summary()
'''
    
    print(example_code)
    return example_code

# Configuration customization example
# print("\n" + "="*50)
# print("CONFIGURATION CUSTOMIZATION EXAMPLE")
# print("="*50)
# print("You can customize the pipeline configuration like this:")
# print("""
# # Custom configuration
# custom_config = {
#     'BATCH_SIZE': 2000,           # Larger batch size
#     'MAX_EPOCHS': 200,            # More epochs
#     'PATIENCE': 20,               # More patience for early stopping
#     'MIN_DELTA': 0.0005,          # Smaller improvement threshold
#     'VALIDATION_SPLIT': 0.15,     # Less validation data
#     'TOP_K_CLASSES': 50,          # Fewer classes for ROC AUC
#     'ENABLE_MEMORY_PROFILING': False,  # Disable for speed
#     'BASE_DIR': '/custom/path/'   # Custom output directory
# }

# # Use custom config
# from enhanced_ml_pipeline import EnhancedMLPipeline
# pipeline = EnhancedMLPipeline(custom_config)

# # Then use individual pipeline methods
# model = SGDClassifier(...)
# trained_model = pipeline.fit_model_safely(model, X_train, y_train, level, method)
# """)

# If you want to customize the configuration:
# custom_config = CONFIG.copy()
# custom_config['MAX_EPOCHS'] = 200
# custom_config['PATIENCE'] = 20
# custom_config['BATCH_SIZE'] = 2000
#
# # Then pass it to the pipeline constructor
# from enhanced_ml_pipeline import EnhancedMLPipeline
# pipeline = EnhancedMLPipeline(custom_config)
# # ... use pipeline.fit_model_safely, etc.

# =================================================================
# NAIVE BAYES VALIDATION FOR DIMENSIONALITY REDUCTION RESULTS
# =================================================================

def validate_naive_bayes_simple(csv_path_X_train, csv_path_X_test, 
                               csv_path_y_train, csv_path_y_test,
                               label_encoder_path=None):
    """
    Simple Naive Bayes validation for dimensionality reduction results
    
    Args:
        csv_path_X_train: Path to X_train CSV file (features only)
        csv_path_X_test: Path to X_test CSV file (features only)  
        csv_path_y_train: Path to y_train CSV file
        csv_path_y_test: Path to y_test CSV file
        label_encoder_path: Path to label encoder pickle file (optional)
    
    Returns:
        Dictionary with validation results
    """
    print("🚀 NAIVE BAYES VALIDATION")
    print("="*50)
    
    try:
        # Load data
        print("📁 Loading data...")
        X_train = pd.read_csv(csv_path_X_train).values
        X_test = pd.read_csv(csv_path_X_test).values
        
        # Load labels
        y_train = pd.read_csv(csv_path_y_train)['y_train'].values
        y_test = pd.read_csv(csv_path_y_test)['y_test'].values
        
        print(f"✅ Data loaded successfully:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   X_test shape:  {X_test.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   y_test shape:  {y_test.shape}")
        print(f"   Train classes: {len(np.unique(y_train))}")
        print(f"   Test classes:  {len(np.unique(y_test))}")
        
        # Validate shapes
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Train data mismatch: X_train {X_train.shape[0]} != y_train {y_train.shape[0]}")
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Test data mismatch: X_test {X_test.shape[0]} != y_test {y_test.shape[0]}")
        
        # Train Naive Bayes
        print("\n🤖 Training Naive Bayes classifier...")
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Make predictions
        print("🎯 Making predictions...")
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n📊 RESULTS:")
        print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Load label encoder if available
        label_encoder = None
        if label_encoder_path and os.path.exists(label_encoder_path):
            try:
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                print(f"✅ Label encoder loaded from: {label_encoder_path}")
            except Exception as e:
                print(f"⚠️ Could not load label encoder: {str(e)}")
        
        # Classification report
        print(f"\n📋 Classification Report (Test Set):")
        from sklearn.metrics import classification_report
        
        if label_encoder:
            try:
                # Convert back to original labels
                y_test_original = label_encoder.inverse_transform(y_test)
                y_test_pred_original = label_encoder.inverse_transform(y_test_pred)
                print(classification_report(y_test_original, y_test_pred_original))
            except:
                print(classification_report(y_test, y_test_pred))
        else:
            print(classification_report(y_test, y_test_pred))
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train)),
            'model': model
        }
        
        print("✅ Validation completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Error in validation: {str(e)}")
        return None

def validate_multiple_methods(base_path="/content/rki_2025", level='species', kmer=6, 
                            methods_configs=None):
    """
    Validate multiple dimensionality reduction methods
    
    Args:
        base_path: Base path to data
        level: Taxonomic level
        kmer: K-mer size
        methods_configs: List of method configurations
    
    Example:
        configs = [
            {'method': 'IPCA', 'device': 'CPU', 'n_components': 68},
            {'method': 'UMAP', 'device': 'CPU', 'n_components': 50},
            {'method': 'SVD', 'device': 'CPU', 'n_components': 100}
        ]
        results = validate_multiple_methods(base_path, 'species', 6, configs)
    """
    print("🚀 MULTIPLE METHODS VALIDATION")
    print("="*60)
    
    if methods_configs is None:
        methods_configs = [
            {'method': 'IPCA', 'device': 'CPU', 'n_components': 68},
            {'method': 'UMAP', 'device': 'CPU', 'n_components': 50}
        ]
    
    all_results = []
    
    for i, config in enumerate(methods_configs, 1):
        print(f"\n📊 Method {i}/{len(methods_configs)}: {config['method']}-{config['device']}")
        print("-"*40)
        
        try:
            method = config['method']
            device = config['device']
            n_components = config['n_components']
            fold = config.get('fold', 0)
            
            # Determine optimization method
            optimization_method = "manifold" if method.lower() in ['umap', 'autoencoder'] else "cev"
            
            # Construct file paths
            reduction_base = f"{base_path}/prep/reduction/{level}/k{kmer}/{method.upper()}/{device.upper()}"
            vectorization_base = f"{base_path}/prep/vectorization/{level}/k{kmer}"
            
            # File paths
            csv_path_X_train = f"{reduction_base}/fold{fold}_X_train_fold{fold}_{optimization_method}_{n_components}components.csv"
            csv_path_X_test = f"{reduction_base}/fold{fold}_X_test_fold{fold}_{optimization_method}_{n_components}components.csv"
            csv_path_y_train = f"{vectorization_base}/y_train_k{kmer}_{level}.csv"
            csv_path_y_test = f"{vectorization_base}/y_test_k{kmer}_{level}.csv"
            label_encoder_path = f"{vectorization_base}/label_encoder_k{kmer}_{level}.pkl"
            
            # Validate
            result = validate_naive_bayes_simple(
                csv_path_X_train, csv_path_X_test,
                csv_path_y_train, csv_path_y_test,
                label_encoder_path
            )
            
            if result:
                result_summary = {
                    'method': method,
                    'device': device,
                    'level': level,
                    'kmer': kmer,
                    'n_components': n_components,
                    'optimization_method': optimization_method,
                    **{k: v for k, v in result.items() if k != 'model'}
                }
                all_results.append(result_summary)
                
        except Exception as e:
            print(f"❌ Error validating {config}: {str(e)}")
            continue
    
    # Summary
    if all_results:
        print(f"\n📊 VALIDATION SUMMARY:")
        print("-"*60)
        print(f"{'Method-Device':<15} | {'Components':<10} | {'Test Acc':<10} | {'Classes':<8}")
        print("-"*60)
        
        for result in all_results:
            method_device = f"{result['method']}-{result['device']}"
            components = result['n_components']
            test_acc = f"{result['test_accuracy']:.4f}"
            classes = result['n_classes']
            print(f"{method_device:<15} | {components:<10} | {test_acc:<10} | {classes:<8}")
        
        # Best performing method
        best_result = max(all_results, key=lambda x: x['test_accuracy'])
        print(f"\n🏆 Best performing method:")
        print(f"   {best_result['method']}-{best_result['device']} with {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%) accuracy")
    
    return all_results

# Example usage for your specific case
def example_validation():
    """Example validation for your specific paths"""
    print("📚 EXAMPLE VALIDATION")
    print("="*40)
    
    # Example 1: Single validation
    print("\n1️⃣ Single Method Validation:")
    
    csv_path_X_train = '/content/rki_2025/prep/reduction/species/k6/IPCA/CPU/fold0_X_train_fold0_cev_68components.csv'
    csv_path_X_test = '/content/rki_2025/prep/reduction/species/k6/IPCA/CPU/fold0_X_test_fold0_cev_68components.csv'
    csv_path_y_train = '/content/rki_2025/prep/vectorization/species/k6/y_train_k6_species.csv'
    csv_path_y_test = '/content/rki_2025/prep/vectorization/species/k6/y_test_k6_species.csv'
    label_encoder_path = '/content/rki_2025/prep/vectorization/species/k6/label_encoder_k6_species.pkl'
    
    result = validate_naive_bayes_simple(
        csv_path_X_train, csv_path_X_test,
        csv_path_y_train, csv_path_y_test,
        label_encoder_path
    )
    
    # Example 2: Multiple methods validation
    print("\n2️⃣ Multiple Methods Validation:")
    
    configs = [
        {'method': 'IPCA', 'device': 'CPU', 'n_components': 68},
        {'method': 'UMAP', 'device': 'CPU', 'n_components': 50},
        {'method': 'SVD', 'device': 'CPU', 'n_components': 100}
    ]
    
    results = validate_multiple_methods(
        base_path="/content/rki_2025", 
        level='species', 
        kmer=6, 
        methods_configs=configs
    )
    
    return result, results


# ================================
# MAIN EXECUTION & EXAMPLES
# ================================

if __name__ == "__main__":
    """
    Contoh eksekusi EnhancedDataLoader dengan pipeline modeling
    """
    
    print("🚀 **ENHANCED DATA LOADER - EXECUTION EXAMPLES**")
    print("="*60)
    
    # Setup paths
    reduction_log_path = "/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction/output_paths_log.txt"
    vectorization_paths_file = "/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization_config/all_vectorization_output_paths.txt"
    
    # 1. Initialize dan explore available datasets
    print("\n1️⃣ **DATASET DISCOVERY**")
    print("-" * 30)
    
    loader = EnhancedDataLoader(reduction_log_path, vectorization_paths_file)
    summary = loader.get_dataset_summary()
    
    # 2. Single dataset example
    print("\n2️⃣ **SINGLE DATASET PROCESSING**")
    print("-" * 30)
    
    # Pilih dataset pertama yang available
    available = loader.discover_available_datasets()
    if available:
        level, kmer = available[0]  # Ambil dataset pertama
        
        results_df, best_model, metadata = run_enhanced_pipeline_with_enhanced_loader(
            level=level,
            kmer=kmer,
            reduction_log_path=reduction_log_path,
            vectorization_paths_file=vectorization_paths_file
        )
        
        if results_df is not None:
            print(f"\n✅ Pipeline completed for {level} {kmer}")
            print(f"Best Model: {best_model}")
            print("\nTop Results:")
            print(results_df.head())
    
    # 3. Batch processing example (commented untuk demo)
    print("\n3️⃣ **BATCH PROCESSING EXAMPLE (Commented)**")
    print("-" * 30)
    print("""
    # Uncomment untuk batch processing:
    
    batch_results = batch_process_with_enhanced_loader(
        reduction_log_path=reduction_log_path,
        vectorization_paths_file=vectorization_paths_file,
        levels=["class", "family"],  # Filter levels
        kmers=["k6", "k8"]           # Filter kmers
    )
    
    # Aggregate results
    all_dfs = []
    for (level, kmer), result_data in batch_results.items():
        df = result_data['results_df'].copy()
        df['dataset_level'] = level
        df['dataset_kmer'] = kmer
        all_dfs.append(df)
    
    if all_dfs:
        combined_results = pd.concat(all_dfs, ignore_index=True)
        combined_results.to_csv('enhanced_loader_batch_results.csv', index=False)
        print(f"Saved combined results: {len(combined_results)} records")
    """)
    
    # 4. Print example usage
    print("\n4️⃣ **EXAMPLE CODE USAGE**")
    print("-" * 30)
    get_enhanced_loader_example_usage()