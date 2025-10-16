import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional imports
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Disable GPU warnings
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CUML_AVAILABLE = False

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =================================================================
# MEMORY MANAGEMENT UTILITIES - NEW!
# =================================================================

class MemoryManager:
    """Advanced memory management untuk large datasets"""
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage"""
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_gb': mem_info.rss / 1e9,
            'available_gb': psutil.virtual_memory().available / 1e9,
            'percent': psutil.virtual_memory().percent
        }
    
    @staticmethod
    def estimate_dense_memory(sparse_matrix):
        """Estimate memory needed if converted to dense"""
        n_rows, n_cols = sparse_matrix.shape
        dense_size_gb = (n_rows * n_cols * 8) / 1e9  # float64
        return dense_size_gb
    
    @staticmethod
    def safe_batch_size(X, target_memory_gb=2.0):
        """Calculate safe batch size based on available memory"""
        mem_info = MemoryManager.get_memory_info()
        available_gb = mem_info['available_gb'] * 0.5  # Use only 50% of available
        
        row_memory_gb = (X.shape[1] * 8) / 1e9  # One row in dense format
        safe_batch = int(min(target_memory_gb / row_memory_gb, X.shape[0]))
        
        return max(100, safe_batch)  # Minimum 100
    
    @staticmethod
    def force_gc():
        """Aggressive garbage collection"""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def log_memory(prefix=""):
        """Log current memory status"""
        mem_info = MemoryManager.get_memory_info()
        print(f"🧠 {prefix} Memory: {mem_info['rss_gb']:.2f}GB used, "
              f"{mem_info['available_gb']:.2f}GB available ({mem_info['percent']:.1f}%)")

# =================================================================
# 1. CONFIGURATION CLASS
# =================================================================

class BenchmarkConfig:
    """Unified configuration untuk benchmark"""
    def __init__(
        self,
        # Data parameters
        levels: List[str] = ['genus', 'species'],
        kmers: List[int] = [6],
        
        # Method & device
        methods: List[str] = ['ipca', 'svd'],
        devices: List[str] = ['cpu'],
        
        # Optimization parameters
        cev_threshold: float = 0.95,
        start_components: int = 100,
        step_components: int = 50,
        max_components: int = 1000,
        
        # Processing parameters - MEMORY SAFE DEFAULTS
        batch_size: int = None,  # Auto-calculate
        cv_folds: int = 2,
        max_memory_gb: float = 4.0,  # Max memory per operation
        
        # Output parameters
        output_dir: str = None,
        skip_existing: bool = True,
        
        # Plotting parameters
        plot_format: str = 'png',
        plot_dpi: int = 300,
        create_plots: bool = True,
        plot_sample_size: int = 5000,  # Max points to plot
        
        # Optional parameters
        autoencoder_epochs: int = 30,
        umap_neighbors: int = 15,
        
        # Safety parameters
        enable_memory_monitoring: bool = True,
        emergency_stop_threshold: float = 90.0  # Stop if memory > 90%
    ):
        self.levels = levels
        self.kmers = kmers
        self.methods = methods
        self.devices = devices
        self.cev_threshold = cev_threshold
        self.start_components = start_components
        self.step_components = step_components
        self.max_components = max_components
        self.batch_size = batch_size
        self.cv_folds = cv_folds
        self.max_memory_gb = max_memory_gb
        self.output_dir = output_dir or '/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction'
        self.skip_existing = skip_existing
        self.plot_format = plot_format
        self.plot_dpi = plot_dpi
        self.create_plots = create_plots
        self.plot_sample_size = plot_sample_size
        self.autoencoder_epochs = autoencoder_epochs
        self.umap_neighbors = umap_neighbors
        self.enable_memory_monitoring = enable_memory_monitoring
        self.emergency_stop_threshold = emergency_stop_threshold

# =================================================================
# 2. DATA LOADER WITH MEMORY CHECKS
# =================================================================

class DataLoader:
    """Memory-safe data loader untuk RKI struktur"""
    def __init__(self, base_path: str, config: BenchmarkConfig):
        self.base_path = Path(base_path)
        self.config = config
        
    def load_data(self, level: str, kmer: int, split: str = 'train'):
        """Load data dengan memory monitoring"""
        MemoryManager.log_memory(f"Before loading {level} k{kmer}")
        
        level_dir = self.base_path / level / f"k{kmer}"
        
        if split == 'train':
            X_file = level_dir / f"X_train_sparse_k{kmer}_{level}.npz"
            y_file = level_dir / f"y_train_k{kmer}_{level}.npy"
        else:
            X_file = level_dir / f"X_test_sparse_k{kmer}_{level}.npz"
            y_file = level_dir / f"y_test_k{kmer}_{level}.npy"
        
        # Check if files exist
        if not X_file.exists() or not y_file.exists():
            raise FileNotFoundError(f"Data files not found: {X_file}")
        
        # Load sparse matrix
        X_data = np.load(X_file)
        X = sparse.csr_matrix(
            (X_data['data'], X_data['indices'], X_data['indptr']), 
            shape=X_data['shape']
        )
        
        # Load labels
        y = np.load(y_file)
        
        # Estimate memory requirement
        dense_size = MemoryManager.estimate_dense_memory(X)
        print(f"📊 Data shape: {X.shape}")
        print(f"💾 Sparse size: {X.data.nbytes / 1e9:.2f}GB")
        print(f"⚠️  Dense would be: {dense_size:.2f}GB")
        
        # Check if batch processing is needed
        if dense_size > self.config.max_memory_gb:
            print(f"⚠️  WARNING: Data too large for single batch!")
            print(f"   Will use incremental processing")
        
        MemoryManager.log_memory(f"After loading {level} k{kmer}")
        
        return X, y

# =================================================================
# 3. PLOTTING MANAGER WITH MEMORY SAFETY
# =================================================================

class PlottingManager:
    """Memory-safe plotting functionality"""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.plot_format = config.plot_format
        self.dpi = config.plot_dpi
        self.sample_size = config.plot_sample_size
        
    def _safe_sample(self, X, y, max_samples=None):
        """Safely sample data for plotting"""
        if max_samples is None:
            max_samples = self.sample_size
            
        n_samples = min(max_samples, len(X))
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            return X[indices], y[indices]
        return X, y
    
    def plot_variance_explained(self, variance_ratio, optimal_n, level, kmer, method, output_dir):
        """Plot cumulative explained variance"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cev = np.cumsum(variance_ratio)
            ax.plot(range(1, len(cev) + 1), cev, 'b-', linewidth=2, label='Cumulative Variance')
            ax.axhline(y=self.config.cev_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({self.config.cev_threshold:.2f})')
            ax.axvline(x=optimal_n, color='g', linestyle='--', 
                       label=f'Optimal Components ({optimal_n})')
            
            ax.set_xlabel('Number of Components', fontsize=12)
            ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
            ax.set_title(f'Variance Explained - {level.upper()} K{kmer} ({method.upper()})', 
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            filename = f"variance_explained.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            MemoryManager.force_gc()
            return str(filepath)
        except Exception as e:
            print(f"⚠️  Failed to create variance plot: {e}")
            plt.close('all')
            return None
    
    def plot_2d_projection(self, X_transformed, y, level, kmer, method, output_dir):
        """Plot 2D projection dengan memory safety"""
        try:
            if X_transformed.shape[1] < 2:
                return None
            
            # Safe sampling
            X_plot, y_plot = self._safe_sample(X_transformed, y)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot by class
            unique_classes = np.unique(y_plot)[:20]  # Max 20 classes in legend
            for cls in unique_classes:
                mask = y_plot == cls
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                          alpha=0.6, s=30, label=f'Class {cls}')
            
            ax.set_xlabel('Component 1', fontsize=12)
            ax.set_ylabel('Component 2', fontsize=12)
            ax.set_title(f'2D Projection - {level.upper()} K{kmer} ({method.upper()})\n'
                        f'(Showing {len(X_plot)}/{len(X_transformed)} samples)', 
                         fontsize=14, fontweight='bold')
            
            if len(unique_classes) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            ax.grid(True, alpha=0.3)
            
            filename = f"projection_2d.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            MemoryManager.force_gc()
            return str(filepath)
        except Exception as e:
            print(f"⚠️  Failed to create 2D projection: {e}")
            plt.close('all')
            return None
    
    def plot_component_distribution(self, X_transformed, level, kmer, method, output_dir):
        """Plot distribution dengan memory safety"""
        try:
            n_comp_to_plot = min(5, X_transformed.shape[1])
            
            # Safe sampling
            X_plot, _ = self._safe_sample(X_transformed, np.zeros(len(X_transformed)))
            
            fig, axes = plt.subplots(1, n_comp_to_plot, figsize=(4*n_comp_to_plot, 4))
            if n_comp_to_plot == 1:
                axes = [axes]
            
            for i in range(n_comp_to_plot):
                axes[i].hist(X_plot[:, i], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(f'Component {i+1}', fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].set_title(f'PC{i+1} Distribution', fontsize=11)
                axes[i].grid(True, alpha=0.3)
            
            fig.suptitle(f'Component Distributions - {level.upper()} K{kmer} ({method.upper()})', 
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            filename = f"component_distributions.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            MemoryManager.force_gc()
            return str(filepath)
        except Exception as e:
            print(f"⚠️  Failed to create distribution plot: {e}")
            plt.close('all')
            return None
    
    def plot_summary_comparison(self, results_df, output_dir):
        """Plot summary comparison"""
        try:
            if len(results_df) == 0:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Components by method
            ax = axes[0, 0]
            results_df.groupby('method')['n_components'].mean().plot(kind='bar', ax=ax)
            ax.set_title('Average Components by Method', fontsize=12, fontweight='bold')
            ax.set_xlabel('Method')
            ax.set_ylabel('Number of Components')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 2. Silhouette score by method
            ax = axes[0, 1]
            results_df.groupby('method')['silhouette'].mean().plot(kind='bar', ax=ax, color='green')
            ax.set_title('Average Silhouette Score by Method', fontsize=12, fontweight='bold')
            ax.set_xlabel('Method')
            ax.set_ylabel('Silhouette Score')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 3. Time by method
            ax = axes[1, 0]
            results_df.groupby('method')['time'].mean().plot(kind='bar', ax=ax, color='red')
            ax.set_title('Average Time by Method', fontsize=12, fontweight='bold')
            ax.set_xlabel('Method')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. Heatmap: Silhouette by level x kmer
            ax = axes[1, 1]
            if 'level' in results_df.columns and 'kmer' in results_df.columns:
                pivot = results_df.pivot_table(values='silhouette', 
                                              index='level', 
                                              columns='kmer', 
                                              aggfunc='mean')
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
                ax.set_title('Silhouette Score: Level x K-mer', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            filename = f"summary_comparison.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            MemoryManager.force_gc()
            return str(filepath)
        except Exception as e:
            print(f"⚠️  Failed to create summary plot: {e}")
            plt.close('all')
            return None

# =================================================================
# 4. OUTPUT MANAGER
# =================================================================

class OutputManager:
    """Simplified output manager"""
    def __init__(self, base_path: str, skip_existing: bool = True):
        self.base_path = Path(base_path)
        self.skip_existing = skip_existing
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def get_output_dir(self, level: str, kmer: int, method: str):
        """Get organized output directory"""
        output_dir = self.base_path / level / f"k{kmer}" / method
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        return self.skip_existing and file_path.exists()
    
    def save_file(self, content, filename: str, level: str, kmer: int, method: str):
        """Save file to organized directory"""
        output_dir = self.get_output_dir(level, kmer, method)
        file_path = output_dir / filename
        
        if self.should_skip(file_path):
            print(f"⏭️  Skipping: {file_path}")
            return None
        
        if hasattr(content, 'to_csv'):
            content.to_csv(file_path, index=False)
        elif hasattr(content, 'savefig'):
            content.savefig(file_path, dpi=300, bbox_inches='tight')
        else:
            np.save(file_path, content)
        
        print(f"💾 Saved: {file_path}")
        return str(file_path)

# =================================================================
# 5. MEMORY-SAFE BENCHMARK CLASS
# =================================================================

class SimplifiedBenchmark:
    """Memory-optimized benchmark class"""
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_mgr = OutputManager(config.output_dir, config.skip_existing)
        self.plotter = PlottingManager(config) if config.create_plots else None
        self.results = []
        
    def _check_memory_emergency(self):
        """Check if we need to stop due to memory"""
        if self.config.enable_memory_monitoring:
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > self.config.emergency_stop_threshold:
                raise MemoryError(
                    f"Emergency stop! Memory usage: {mem_percent:.1f}% > "
                    f"{self.config.emergency_stop_threshold}%"
                )
    
    def find_optimal_components(self, X, y, method):
        """Find optimal components dengan memory management"""
        MemoryManager.log_memory("Before component optimization")
        
        n_comp = self.config.start_components
        
        # Auto-calculate batch size if not provided
        if self.config.batch_size is None:
            batch_size = MemoryManager.safe_batch_size(X, self.config.max_memory_gb)
            print(f"🔧 Auto batch size: {batch_size}")
        else:
            batch_size = self.config.batch_size
        
        if method == 'ipca':
            model = IncrementalPCA(n_components=n_comp)
            
            # Incremental fitting dengan progress bar
            n_batches = int(np.ceil(X.shape[0] / batch_size))
            pbar = tqdm(total=n_batches, desc="Fitting IncrementalPCA")
            
            for i in range(0, X.shape[0], batch_size):
                self._check_memory_emergency()
                
                batch = X[i:i+batch_size].toarray()
                model.partial_fit(batch)
                
                # Clean up
                del batch
                MemoryManager.force_gc()
                pbar.update(1)
            
            pbar.close()
            variance_ratio = model.explained_variance_ratio_
            cev = np.cumsum(variance_ratio)
            
        elif method == 'svd':
            # SVD tidak bisa incremental, tapi bisa sparse
            print("⚠️  SVD uses full matrix, checking memory...")
            dense_size = MemoryManager.estimate_dense_memory(X)
            
            if dense_size > self.config.max_memory_gb * 2:
                print(f"⚠️  WARNING: Data might be too large for SVD")
                print(f"   Consider using IPCA instead or reducing n_components")
            
            model = TruncatedSVD(n_components=n_comp)
            model.fit(X)
            variance_ratio = model.explained_variance_ratio_
            cev = np.cumsum(variance_ratio)
        
        else:
            return n_comp, None
        
        if np.any(cev >= self.config.cev_threshold):
            optimal_n = np.argmax(cev >= self.config.cev_threshold) + 1
        else:
            optimal_n = n_comp
        
        MemoryManager.log_memory("After component optimization")
        MemoryManager.force_gc()
        
        return optimal_n, variance_ratio
    
    def transform_data(self, X, y, method, n_components):
        """Transform data dengan memory management"""
        MemoryManager.log_memory("Before transformation")
        
        # Auto-calculate batch size
        if self.config.batch_size is None:
            batch_size = MemoryManager.safe_batch_size(X, self.config.max_memory_gb)
        else:
            batch_size = self.config.batch_size
        
        if method == 'ipca':
            model = IncrementalPCA(n_components=n_components)
            
            # Fit incrementally
            print("🔧 Fitting model...")
            n_batches = int(np.ceil(X.shape[0] / batch_size))
            pbar = tqdm(total=n_batches, desc="Fitting")
            
            for i in range(0, X.shape[0], batch_size):
                self._check_memory_emergency()
                batch = X[i:i+batch_size].toarray()
                model.partial_fit(batch)
                del batch
                MemoryManager.force_gc()
                pbar.update(1)
            pbar.close()
            
            # Transform incrementally
            print("🔧 Transforming data...")
            X_transformed = []
            pbar = tqdm(total=n_batches, desc="Transforming")
            
            for i in range(0, X.shape[0], batch_size):
                self._check_memory_emergency()
                batch = X[i:i+batch_size].toarray()
                X_transformed.append(model.transform(batch))
                del batch
                MemoryManager.force_gc()
                pbar.update(1)
            pbar.close()
            
            result = np.vstack(X_transformed)
            del X_transformed
            MemoryManager.force_gc()
            
            return result
            
        elif method == 'svd':
            print("🔧 Transforming with SVD (using sparse matrix)...")
            model = TruncatedSVD(n_components=n_components)
            result = model.fit_transform(X)
            MemoryManager.force_gc()
            return result
            
        elif method == 'umap' and UMAP_AVAILABLE:
            print("⚠️  UMAP requires dense matrix - using sampling if needed")
            
            # Check if we need to sample
            dense_size = MemoryManager.estimate_dense_memory(X)
            if dense_size > self.config.max_memory_gb:
                max_samples = int((self.config.max_memory_gb * 1e9) / (X.shape[1] * 8))
                print(f"⚠️  Sampling to {max_samples} rows for UMAP")
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X_sample = X[indices].toarray()
                y_sample = y[indices]
            else:
                X_sample = X.toarray()
                y_sample = y
            
            model = umap.UMAP(
                n_components=n_components, 
                n_neighbors=self.config.umap_neighbors
            )
            result = model.fit_transform(X_sample)
            
            del X_sample
            MemoryManager.force_gc()
            return result
        
        return None
    
    def save_results(self, X_transformed, y, method, level, kmer, fold):
        """Save results dengan chunking untuk memory safety"""
        MemoryManager.log_memory("Before saving results")
        
        output_dir = self.output_mgr.get_output_dir(level, kmer, method)
        filename = f"features_fold{fold}.csv"
        filepath = output_dir / filename
        
        if self.output_mgr.should_skip(filepath):
            print(f"⏭️  Skipping: {filepath}")
            return
        
        # Save in chunks if data is large
        chunk_size = 10000
        if len(X_transformed) > chunk_size:
            print(f"💾 Saving in chunks...")
            
            # Save first chunk with header
            df_chunk = pd.DataFrame(X_transformed[:chunk_size])
            df_chunk['label'] = y[:chunk_size]
            df_chunk.to_csv(filepath, index=False, mode='w')
            
            # Append remaining chunks
            for i in range(chunk_size, len(X_transformed), chunk_size):
                df_chunk = pd.DataFrame(X_transformed[i:i+chunk_size])
                df_chunk['label'] = y[i:i+chunk_size]
                df_chunk.to_csv(filepath, index=False, mode='a', header=False)
                del df_chunk
                MemoryManager.force_gc()
        else:
            df = pd.DataFrame(X_transformed)
            df['label'] = y
            df.to_csv(filepath, index=False)
        
        print(f"💾 Saved: {filepath}")
        MemoryManager.log_memory("After saving results")
    
    def run(self, loader: DataLoader):
        """Main benchmark loop dengan comprehensive memory management"""
        print("🚀 Starting Memory-Safe Benchmark")
        print(f"⚙️  Max memory per operation: {self.config.max_memory_gb}GB")
        print(f"⚙️  Emergency stop threshold: {self.config.emergency_stop_threshold}%")
        
        MemoryManager.log_memory("Initial")
        
        for level in self.config.levels:
            for kmer in self.config.kmers:
                print(f"\n{'='*60}")
                print(f"🧬 {level.upper()} - K-mer {kmer}")
                
                try:
                    X_train, y_train = loader.load_data(level, kmer, 'train')
                    print(f"✅ Data loaded: {X_train.shape}")
                    
                    for method in self.config.methods:
                        print(f"\n🔧 Method: {method.upper()}")
                        
                        try:
                            start = time.time()
                            
                            # Find optimal components
                            n_comp, variance_ratio = self.find_optimal_components(
                                X_train, y_train, method
                            )
                            print(f"   Optimal components: {n_comp}")
                            
                            # Transform data
                            X_transformed = self.transform_data(
                                X_train, y_train, method, n_comp
                            )
                            
                            # Calculate silhouette on sample
                            sample_size = min(1000, len(X_transformed))
                            indices = np.random.choice(len(X_transformed), sample_size, replace=False)
                            silhouette = silhouette_score(
                                X_transformed[indices],
                                y_train[indices]
                            )
                            
                            elapsed = time.time() - start
                            print(f"   Silhouette (sample): {silhouette:.4f}")
                            print(f"   Time: {elapsed:.2f}s")
                            
                            # Save results
                            self.save_results(X_transformed, y_train, method, level, kmer, 0)
                            
                            # Create plots
                            if self.plotter:
                                output_dir = self.output_mgr.get_output_dir(level, kmer, method)
                                print(f"\n📊 Creating plots...")
                                
                                if variance_ratio is not None:
                                    self.plotter.plot_variance_explained(
                                        variance_ratio, n_comp, level, kmer, method, output_dir
                                    )
                                
                                self.plotter.plot_2d_projection(
                                    X_transformed, y_train, level, kmer, method, output_dir
                                )
                                
                                self.plotter.plot_component_distribution(
                                    X_transformed, level, kmer, method, output_dir
                                )
                            
                            self.results.append({
                                'level': level,
                                'kmer': kmer,
                                'method': method,
                                'n_components': n_comp,
                                'silhouette': silhouette,
                                'time': elapsed
                            })
                            
                            # Clean up
                            del X_transformed
                            MemoryManager.force_gc()
                            MemoryManager.log_memory(f"After {method}")
                            
                        except MemoryError as e:
                            print(f"❌ MEMORY ERROR: {e}")
                            print(f"   Skipping {method} for {level} k{kmer}")
                            MemoryManager.force_gc()
                            continue
                        
                        except Exception as e:
                            print(f"❌ ERROR: {e}")
                            print(f"   Skipping {method} for {level} k{kmer}")
                            continue
                    
                    # Clean up after each dataset
                    del X_train, y_train
                    MemoryManager.force_gc()
                    
                except Exception as e:
                    print(f"❌ ERROR loading {level} k{kmer}: {e}")
                    continue
        
        # Save summary
        summary_df = pd.DataFrame(self.results)
        summary_file = self.output_mgr.base_path / f"summary_{int(time.time())}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary: {summary_file}")
        
        # Create summary plot
        if self.plotter and len(summary_df) > 0:
            print(f"\n📊 Creating summary comparison plot...")
            self.plotter.plot_summary_comparison(summary_df, self.output_mgr.base_path)
        
        MemoryManager.log_memory("Final")
        
        return summary_df

# =================================================================
# 6. MEMORY-SAFE API
# =================================================================

def run_benchmark(
    data_path: str,
    levels: List[str] = ['genus', 'species'],
    kmers: List[int] = [6],
    methods: List[str] = ['ipca', 'svd'],
    output_dir: str = None,
    create_plots: bool = True,
    max_memory_gb: float = 4.0,
    **kwargs
):
    """
    🚀 MEMORY-SAFE BENCHMARK API
    
    Args:
        data_path: Path ke vectorization directory
        levels: List taxonomic levels
        kmers: List k-mer sizes
        methods: List methods ('ipca', 'svd', 'umap')
        output_dir: Output directory
        create_plots: Whether to create plots
        max_memory_gb: Maximum memory per operation (GB)
        **kwargs: Additional parameters
        
    Returns:
        DataFrame dengan hasil benchmark
        
    Example for large datasets:
        results = run_benchmark(
            data_path='/path/to/data',
            levels=['genus'],
            kmers=[6],
            methods=['ipca'],  # Recommended for large data
            max_memory_gb=4.0,  # Adjust based on available RAM
            batch_size=None,  # Auto-calculate
            emergency_stop_threshold=85.0
        )
    """
    config = BenchmarkConfig(
        levels=levels,
        kmers=kmers,
        methods=methods,
        output_dir=output_dir,
        create_plots=create_plots,
        max_memory_gb=max_memory_gb,
        **kwargs
    )
    
    loader = DataLoader(data_path, config)
    benchmark = SimplifiedBenchmark(config)
    
    return benchmark.run(loader)

# =================================================================
# USAGE EXAMPLES
# =================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MEMORY-SAFE BENCHMARK - USAGE EXAMPLES")
    print("=" * 80)
    
    # Example 1: Large dataset (recommended)
    print("\n📝 Example 1: Large dataset (400K rows x 60K features)")
    print("""
    results = run_benchmark(
        data_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization',
        levels=['genus'],
        kmers=[6],
        methods=['ipca'],  # IPCA is memory-efficient
        max_memory_gb=4.0,  # Max 4GB per operation
        batch_size=None,  # Auto-calculate based on available memory
        emergency_stop_threshold=85.0,  # Stop if memory > 85%
        create_plots=True
    )
    """)
    
    # Example 2: If you have more RAM
    print("\n📝 Example 2: If you have 16GB+ RAM")
    print("""
    results = run_benchmark(
        data_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization',
        levels=['genus'],
        kmers=[6],
        methods=['ipca', 'svd'],
        max_memory_gb=8.0,  # Can use more memory
        batch_size=1000,  # Larger batch size
        create_plots=True
    )
    """)
    
    print("\n✅ Key memory-safe features:")
    print("   ✓ Auto batch size calculation")
    print("   ✓ Incremental processing for IPCA")
    print("   ✓ Memory monitoring at each step")
    print("   ✓ Emergency stop if memory > threshold")
    print("   ✓ Aggressive garbage collection")
    print("   ✓ Chunked file saving")
    print("   ✓ Sampled plotting")
    print("\n⚠️  For 400K x 60K data, recommend:")
    print("   - Use IPCA (not SVD)")
    print("   - max_memory_gb=4.0")
    print("   - batch_size=None (auto)")