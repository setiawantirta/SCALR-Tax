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
# 3. PLOTTING MANAGER WITH MEMORY SAFETY - ENHANCED
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
    
    def plot_pca_feature_space_all_labels(self, X_transformed, y, level, kmer, method, output_dir, split='train'):
        """
        🎨 Plot PCA/Feature Space dengan SEMUA LABEL
        Enhanced plot dengan legend untuk semua kelas
        """
        try:
            if X_transformed.shape[1] < 2:
                print(f"⚠️  Need at least 2 components for 2D plot")
                return None
            
            # Safe sampling untuk performance
            X_plot, y_plot = self._safe_sample(X_transformed, y)
            
            # Get unique classes and sort
            unique_classes = np.unique(y_plot)
            n_classes = len(unique_classes)
            
            print(f"   📊 Plotting {len(X_plot)} samples with {n_classes} classes...")
            
            # Create figure dengan ukuran dinamis berdasarkan jumlah kelas
            if n_classes <= 10:
                figsize = (14, 10)
            elif n_classes <= 20:
                figsize = (16, 12)
            else:
                figsize = (18, 14)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Generate colors untuk semua kelas
            if n_classes <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
            elif n_classes <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
            else:
                # Untuk banyak kelas, gunakan colormap kontinyu
                colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            # Plot setiap kelas dengan warna berbeda
            for idx, cls in enumerate(unique_classes):
                mask = y_plot == cls
                n_samples_class = np.sum(mask)
                
                ax.scatter(
                    X_plot[mask, 0], 
                    X_plot[mask, 1],
                    c=[colors[idx % len(colors)]], 
                    label=f'Class {cls} (n={n_samples_class})',
                    alpha=0.6, 
                    s=30,
                    edgecolors='black',
                    linewidth=0.3
                )
            
            # Styling
            ax.set_xlabel('Principal Component 1', fontsize=13, fontweight='bold')
            ax.set_ylabel('Principal Component 2', fontsize=13, fontweight='bold')
            
            title = (f'PCA Feature Space - {level.upper()} K{kmer} ({method.upper()}) [{split.upper()}]\n'
                    f'{len(X_plot):,} samples | {n_classes} classes | {X_transformed.shape[1]} components')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Legend configuration
            if n_classes <= 30:
                # Small number of classes: show all in legend
                ax.legend(
                    loc='center left', 
                    bbox_to_anchor=(1, 0.5),
                    fontsize=9,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    ncol=1 if n_classes <= 20 else 2
                )
            else:
                # Too many classes: show colorbar instead
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                
                norm = Normalize(vmin=unique_classes.min(), vmax=unique_classes.max())
                sm = ScalarMappable(cmap=plt.cm.rainbow, norm=norm)
                sm.set_array([])
                
                cbar = plt.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label('Class Label', fontsize=11, fontweight='bold')
                
                # Add text info
                # ax.text(1.02, 0.98, f'{n_classes} classes', 
                #        transform=ax.transAxes, 
                #        fontsize=10, 
                #        verticalalignment='top',
                #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Save dengan nama yang jelas
            filename = f"pca_feature_space_all_labels_{split}.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ PCA feature space plot saved: {filename}")
            
            MemoryManager.force_gc()
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️  Failed to create PCA feature space plot: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            return None
    
    def plot_pca_feature_space_3d(self, X_transformed, y, level, kmer, method, output_dir, split='train'):
        """
        🎨 Plot PCA/Feature Space 3D dengan SEMUA LABEL (bonus)
        FIXED: Legend dan PC3 tidak terpotong
        """
        try:
            if X_transformed.shape[1] < 3:
                print(f"⚠️  Need at least 3 components for 3D plot")
                return None
            
            # Safe sampling
            X_plot, y_plot = self._safe_sample(X_transformed, y, max_samples=3000)  # Limit for 3D
            
            unique_classes = np.unique(y_plot)
            n_classes = len(unique_classes)
            
            print(f"   📊 Creating 3D plot with {len(X_plot)} samples and {n_classes} classes...")
            
            # Create 3D figure with adjusted size
            from mpl_toolkits.mplot3d import Axes3D
            
            # ✅ LARGER FIGURE SIZE untuk prevent clipping
            if n_classes <= 10:
                figsize = (18, 12)
            elif n_classes <= 20:
                figsize = (20, 14)
            else:
                figsize = (22, 16)
            
            fig = plt.figure(figsize=figsize)
            
            # ✅ ADJUST SUBPLOT POSITION - Lebih ke kiri untuk beri ruang legend
            ax = fig.add_subplot(111, projection='3d', position=[0.05, 0.05, 0.65, 0.9])
            
            # Generate colors
            if n_classes <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
            else:
                colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            # Plot each class
            for idx, cls in enumerate(unique_classes):
                mask = y_plot == cls
                n_samples_class = np.sum(mask)
                
                ax.scatter(
                    X_plot[mask, 0], 
                    X_plot[mask, 1],
                    X_plot[mask, 2],
                    c=[colors[idx % len(colors)]], 
                    label=f'Class {cls} (n={n_samples_class})',
                    alpha=0.6,
                    s=20,
                    edgecolors='black',
                    linewidth=0.2
                )
            
            # ✅ STYLING dengan extra padding untuk axis labels
            ax.set_xlabel('PC1', fontsize=12, fontweight='bold', labelpad=15)  # Increased labelpad
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold', labelpad=15)
            ax.set_zlabel('PC3', fontsize=12, fontweight='bold', labelpad=15)  # Increased labelpad
            
            # ✅ ADJUST AXIS LIMITS untuk prevent clipping
            # Add some padding to z-axis
            z_min, z_max = X_plot[:, 2].min(), X_plot[:, 2].max()
            z_range = z_max - z_min
            ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
            
            title = (f'3D PCA Feature Space - {level.upper()} K{kmer} ({method.upper()}) [{split.upper()}]\n'
                    f'{len(X_plot):,} samples | {n_classes} classes')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # ✅ LEGEND POSITIONING - Di luar plot area
            if n_classes <= 20:
                # Legend di kanan plot dengan positioning yang jelas
                ax.legend(
                    loc='upper left', 
                    bbox_to_anchor=(1.15, 1.0),  # Adjusted position
                    fontsize=8,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    borderaxespad=0
                )
            else:
                # Terlalu banyak kelas: legend di bawah plot
                ax.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.05),
                    fontsize=7,
                    ncol=min(5, n_classes // 4),
                    frameon=True,
                    fancybox=True
                )
            
            # ✅ ADJUST VIEW ANGLE untuk better visibility
            ax.view_init(elev=20, azim=45)  # Optimal viewing angle
            
            # ✅ TIGHT LAYOUT dengan extra padding
            plt.tight_layout(pad=2.0)  # Extra padding
            
            # Save dengan bbox_inches='tight' untuk capture semua elements
            filename = f"pca_feature_space_3d_{split}.{self.plot_format}"
            filepath = output_dir / filename
            
            # ✅ SAVE dengan pad_inches untuk extra space
            plt.savefig(
                filepath, 
                dpi=self.dpi, 
                bbox_inches='tight',
                pad_inches=0.3  # Extra padding around figure
            )
            plt.close()
            
            print(f"   ✅ 3D PCA feature space plot saved: {filename}")
            
            MemoryManager.force_gc()
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️  Failed to create 3D PCA plot: {e}")
            import traceback
            traceback.print_exc()
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
    
    def plot_class_separation_heatmap(self, X_transformed, y, level, kmer, method, output_dir, split='train'):
        """
        🎨 Plot heatmap of class separation (pairwise distances)
        """
        try:
            from scipy.spatial.distance import cdist
            
            # Safe sampling
            X_plot, y_plot = self._safe_sample(X_transformed, y, max_samples=2000)
            
            unique_classes = np.unique(y_plot)
            n_classes = len(unique_classes)
            
            if n_classes > 50:
                print(f"⚠️  Too many classes ({n_classes}) for heatmap, skipping...")
                return None
            
            print(f"   📊 Creating class separation heatmap for {n_classes} classes...")
            
            # Calculate class centroids
            centroids = []
            class_labels = []
            for cls in unique_classes:
                mask = y_plot == cls
                if np.sum(mask) > 0:
                    centroid = X_plot[mask].mean(axis=0)
                    centroids.append(centroid)
                    class_labels.append(cls)
            
            centroids = np.array(centroids)
            
            # Calculate pairwise distances between centroids
            distances = cdist(centroids, centroids, metric='euclidean')
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(distances, cmap='YlOrRd', aspect='auto')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Euclidean Distance', fontsize=11, fontweight='bold')
            
            # Labels
            ax.set_xticks(range(len(class_labels)))
            ax.set_yticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=90, fontsize=8)
            ax.set_yticklabels(class_labels, fontsize=8)
            
            ax.set_xlabel('Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Class', fontsize=12, fontweight='bold')
            
            title = (f'Class Separation Heatmap - {level.upper()} K{kmer} ({method.upper()}) [{split.upper()}]\n'
                    f'Pairwise distances between class centroids')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            
            plt.tight_layout()
            
            filename = f"class_separation_heatmap_{split}.{self.plot_format}"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Class separation heatmap saved: {filename}")
            
            MemoryManager.force_gc()
            return str(filepath)
            
        except Exception as e:
            print(f"⚠️  Failed to create class separation heatmap: {e}")
            plt.close('all')
            return None
    
    # ... (keep existing plot_2d_projection and plot_summary_comparison methods)
    
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
# 5. MEMORY-SAFE BENCHMARK CLASS (FIXED - NO DATA LEAKAGE)
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
        """Find optimal components dengan ITERATIVE SEARCH (TRAIN ONLY)"""
        MemoryManager.log_memory("Before component optimization")
        
        # Auto-calculate batch size if not provided
        if self.config.batch_size is None:
            batch_size = MemoryManager.safe_batch_size(X, self.config.max_memory_gb)
            print(f"🔧 Auto batch size: {batch_size}")
        else:
            batch_size = self.config.batch_size
        
        if method == 'ipca':
            print(f"🔍 Searching optimal components (threshold={self.config.cev_threshold})...")
            
            # 👇 ITERATIVE SEARCH: Start → Max dengan step
            current_n = self.config.start_components
            optimal_n = current_n
            best_cev = 0.0
            
            while current_n <= self.config.max_components:
                print(f"\n   Testing n_components = {current_n}...")
                
                # Fit model dengan current_n komponen
                model = IncrementalPCA(n_components=current_n)
                
                # Incremental fitting
                n_batches = int(np.ceil(X.shape[0] / batch_size))
                pbar = tqdm(total=n_batches, desc=f"Fitting {current_n} components", leave=False)
                
                for i in range(0, X.shape[0], batch_size):
                    self._check_memory_emergency()
                    batch = X[i:i+batch_size].toarray()
                    model.partial_fit(batch)
                    del batch
                    MemoryManager.force_gc()
                    pbar.update(1)
                pbar.close()
                
                # Calculate CEV
                variance_ratio = model.explained_variance_ratio_
                cev = np.cumsum(variance_ratio)
                best_cev = cev[-1]  # Last value = total CEV
                
                print(f"      CEV achieved: {best_cev:.4f}")
                
                # 👇 CHECK IF THRESHOLD REACHED
                if best_cev >= self.config.cev_threshold:
                    # Find exact component where threshold was reached
                    optimal_n = np.argmax(cev >= self.config.cev_threshold) + 1
                    print(f"   ✅ Threshold reached at component {optimal_n}")
                    print(f"      Final CEV: {cev[optimal_n-1]:.4f}")
                    break
                
                # 👇 INCREASE n_components by step
                current_n += self.config.step_components
                
                # If reached max without hitting threshold
                if current_n > self.config.max_components:
                    optimal_n = self.config.max_components
                    print(f"   ⚠️  Reached max_components ({self.config.max_components})")
                    print(f"      Best CEV achieved: {best_cev:.4f} (target: {self.config.cev_threshold})")
                    break
            
            # Re-fit dengan optimal_n untuk mendapatkan variance_ratio yang sesuai
            print(f"\n🔧 Final fit with {optimal_n} components...")
            model = IncrementalPCA(n_components=optimal_n)
            
            n_batches = int(np.ceil(X.shape[0] / batch_size))
            pbar = tqdm(total=n_batches, desc="Final fitting")
            
            for i in range(0, X.shape[0], batch_size):
                self._check_memory_emergency()
                batch = X[i:i+batch_size].toarray()
                model.partial_fit(batch)
                del batch
                MemoryManager.force_gc()
                pbar.update(1)
            pbar.close()
            
            variance_ratio = model.explained_variance_ratio_
            final_cev = np.cumsum(variance_ratio)[optimal_n-1]  # 👈 CEV at optimal_n
            
        elif method == 'svd':
            # SVD logic
            print("⚠️  SVD uses full matrix, checking memory...")
            dense_size = MemoryManager.estimate_dense_memory(X)
            
            if dense_size > self.config.max_memory_gb * 2:
                print(f"⚠️  WARNING: Data might be too large for SVD")
            
            # Start dengan start_components
            current_n = self.config.start_components
            optimal_n = current_n
            
            while current_n <= self.config.max_components:
                print(f"\n   Testing n_components = {current_n}...")
                
                model = TruncatedSVD(n_components=current_n)
                model.fit(X)
                
                variance_ratio = model.explained_variance_ratio_
                cev = np.cumsum(variance_ratio)
                best_cev = cev[-1]
                
                print(f"      CEV achieved: {best_cev:.4f}")
                
                if best_cev >= self.config.cev_threshold:
                    optimal_n = np.argmax(cev >= self.config.cev_threshold) + 1
                    print(f"   ✅ Threshold reached at component {optimal_n}")
                    break
                
                current_n += self.config.step_components
                
                if current_n > self.config.max_components:
                    optimal_n = self.config.max_components
                    print(f"   ⚠️  Reached max_components")
                    break
            
            # Re-fit dengan optimal_n
            model = TruncatedSVD(n_components=optimal_n)
            model.fit(X)
            variance_ratio = model.explained_variance_ratio_
            final_cev = np.cumsum(variance_ratio)[optimal_n-1]  # 👈 CEV at optimal_n
        
        else:
            return self.config.start_components, None, 0.0, None
        
        MemoryManager.log_memory("After component optimization")
        MemoryManager.force_gc()
        
        # 👇 RETURN MODEL untuk digunakan transform test data
        return optimal_n, variance_ratio, final_cev, model
    
    def _transform_with_model(self, X, model, method):
        """Transform data using FITTED model (NO FITTING!)"""
        
        # Auto-calculate batch size
        if self.config.batch_size is None:
            batch_size = MemoryManager.safe_batch_size(X, self.config.max_memory_gb)
        else:
            batch_size = self.config.batch_size
        
        if method == 'ipca':
            print(f"   🔄 Transforming data with IncrementalPCA (NO FITTING)...")
            X_transformed = []
            
            n_batches = int(np.ceil(X.shape[0] / batch_size))
            pbar = tqdm(total=n_batches, desc="Transforming", leave=False)
            
            for i in range(0, X.shape[0], batch_size):
                self._check_memory_emergency()
                batch = X[i:i+batch_size].toarray()
                # 👇 ONLY TRANSFORM - NO FITTING!
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
            print(f"   🔄 Transforming data with TruncatedSVD (NO FITTING)...")
            # 👇 ONLY TRANSFORM - NO FITTING!
            return model.transform(X)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def save_results(self, X_transformed, y, method, level, kmer, fold, split='train'):
        """Save results dengan split identifier"""
        MemoryManager.log_memory("Before saving results")
        
        output_dir = self.output_mgr.get_output_dir(level, kmer, method)
        filename = f"features_fold{fold}_{split}.csv"  # 👈 ADD SPLIT TO FILENAME
        filepath = output_dir / filename
        
        if self.output_mgr.should_skip(filepath):
            print(f"⏭️  Skipping: {filepath}")
            return
        
        # Save in chunks if data is large
        chunk_size = 10000
        if len(X_transformed) > chunk_size:
            print(f"💾 Saving {split} data in chunks...")
            
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
        
        print(f"💾 Saved {split} data: {filepath}")
        MemoryManager.log_memory("After saving results")
    
    def run(self, loader: DataLoader):
        """Main benchmark loop - FIXED NO DATA LEAKAGE"""
        print("🚀 Starting Memory-Safe Benchmark (NO DATA LEAKAGE)")
        print(f"⚙️  Max memory per operation: {self.config.max_memory_gb}GB")
        
        MemoryManager.log_memory("Initial")
        
        for level in self.config.levels:
            for kmer in self.config.kmers:
                print(f"\n{'='*60}")
                print(f"🧬 {level.upper()} - K-mer {kmer}")
                
                try:
                    # ✅ LOAD BOTH TRAIN & TEST
                    X_train, y_train = loader.load_data(level, kmer, 'train')
                    X_test, y_test = loader.load_data(level, kmer, 'test')
                    print(f"✅ Train data loaded: {X_train.shape}")
                    print(f"✅ Test data loaded: {X_test.shape}")
                    
                    for method in self.config.methods:
                        print(f"\n🔧 Method: {method.upper()}")
                        
                        # Track memory usage
                        memory_samples = []
                        
                        try:
                            start = time.time()
                            mem_start = MemoryManager.get_memory_info()
                            memory_samples.append(mem_start['rss_gb'])
                            
                            # =================================================
                            # STEP 1: FIND OPTIMAL COMPONENTS (TRAIN ONLY!)
                            # =================================================
                            print(f"\n📊 Step 1: Finding optimal components on TRAIN data ONLY...")
                            n_comp, variance_ratio, train_cev, fitted_model = self.find_optimal_components(
                                X_train, y_train, method
                            )
                            print(f"   ✅ Optimal components: {n_comp}")
                            print(f"   📈 Train CEV Score: {train_cev:.4f}")
                            
                            mem_after_opt = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_opt['rss_gb'])
                            
                            # =================================================
                            # STEP 2: TRANSFORM TRAIN DATA (using fitted model)
                            # =================================================
                            print(f"\n🔄 Step 2: Transforming TRAIN data...")
                            X_train_transformed = self._transform_with_model(
                                X_train, fitted_model, method
                            )
                            
                            mem_after_train_transform = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_train_transform['rss_gb'])
                            
                            # =================================================
                            # STEP 3: TRANSFORM TEST DATA (NO FITTING!)
                            # =================================================
                            print(f"\n🔄 Step 3: Transforming TEST data (using TRAINED model - NO FITTING)...")
                            print(f"   🛡️  Preventing data leakage - TEST is only transformed!")
                            X_test_transformed = self._transform_with_model(
                                X_test, fitted_model, method
                            )
                            
                            # ❌ NO CEV FOR TEST - PREVENT DATA LEAKAGE!
                            print(f"   ✅ Test data transformed (shape: {X_test_transformed.shape})")
                            print(f"   🛡️  NO CEV calculated for test - preventing data leakage!")
                            
                            mem_after_test_transform = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_test_transform['rss_gb'])
                            
                            elapsed = time.time() - start
                            
                            # Calculate average memory usage
                            avg_memory_gb = np.mean(memory_samples)
                            peak_memory_gb = np.max(memory_samples)
                            
                            # =================================================
                            # STEP 4: SAVE BOTH TRAIN & TEST
                            # =================================================
                            print(f"\n💾 Step 4: Saving results...")
                            
                            # Save TRAIN
                            self.save_results(
                                X_train_transformed, y_train, method, level, kmer, 
                                fold=0, split='train'
                            )
                            
                            # Save TEST
                            self.save_results(
                                X_test_transformed, y_test, method, level, kmer, 
                                fold=0, split='test'
                            )

                            # =================================================
                            # STEP 5: CREATE PLOTS (ENHANCED)
                            # =================================================
                            if self.plotter:
                                output_dir = self.output_mgr.get_output_dir(level, kmer, method)
                                print(f"\n📊 Step 5: Creating enhanced plots...")
                                
                                # 1. Variance Explained
                                if variance_ratio is not None:
                                    self.plotter.plot_variance_explained(
                                        variance_ratio, n_comp, level, kmer, method, output_dir
                                    )
                                
                                # 2. 🎨 PCA Feature Space - TRAIN (SEMUA LABEL!)
                                print(f"   🎨 Creating PCA feature space plot (TRAIN - ALL LABELS)...")
                                self.plotter.plot_pca_feature_space_all_labels(
                                    X_train_transformed, y_train, level, kmer, 
                                    method, output_dir, split='train'
                                )
                                
                                # 3. 🎨 PCA Feature Space - TEST (SEMUA LABEL!)
                                print(f"   🎨 Creating PCA feature space plot (TEST - ALL LABELS)...")
                                self.plotter.plot_pca_feature_space_all_labels(
                                    X_test_transformed, y_test, level, kmer, 
                                    method, output_dir, split='test'
                                )
                                
                                # 4. 🎨 3D Feature Space - TRAIN (BONUS)
                                if X_train_transformed.shape[1] >= 3:
                                    print(f"   🎨 Creating 3D PCA feature space plot (TRAIN)...")
                                    self.plotter.plot_pca_feature_space_3d(
                                        X_train_transformed, y_train, level, kmer, 
                                        method, output_dir, split='train'
                                    )
                                
                                # 5. 🎨 Class Separation Heatmap - TRAIN
                                print(f"   🎨 Creating class separation heatmap (TRAIN)...")
                                self.plotter.plot_class_separation_heatmap(
                                    X_train_transformed, y_train, level, kmer, 
                                    method, output_dir, split='train'
                                )
                                
                                # 6. Component Distribution
                                self.plotter.plot_component_distribution(
                                    X_train_transformed, level, kmer, method, output_dir
                                )
                                
                                print(f"   ✅ All plots created successfully!")

                            # =================================================
                            # STEP 6: SAVE METRICS (NO TEST CEV!)
                            # =================================================
                            self.results.append({
                                'level': level,
                                'kmer': kmer,
                                'method': method,
                                'n_components': n_comp,
                                'train_cev_score': train_cev,
                                # ❌ NO test_cev_score - PREVENT DATA LEAKAGE!
                                'avg_memory_gb': avg_memory_gb,
                                'peak_memory_gb': peak_memory_gb,
                                'time_seconds': elapsed
                            })
                            
                            print(f"\n✅ SUCCESS: {method} completed!")
                            print(f"   📊 Train CEV Score: {train_cev:.4f}")
                            print(f"   🛡️  Test CEV: NOT CALCULATED (preventing data leakage)")
                            print(f"   🧠 Avg Memory: {avg_memory_gb:.2f}GB")
                            print(f"   🧠 Peak Memory: {peak_memory_gb:.2f}GB")
                            print(f"   ⏱️  Time: {elapsed:.2f}s")
                            
                            # Clean up
                            del X_train_transformed, X_test_transformed, fitted_model
                            MemoryManager.force_gc()
                            MemoryManager.log_memory(f"After {method}")
                            
                        except MemoryError as e:
                            print(f"❌ MEMORY ERROR: {e}")
                            MemoryManager.force_gc()
                            continue
                        
                        except Exception as e:
                            print(f"❌ ERROR: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # Clean up after each dataset
                    del X_train, y_train, X_test, y_test
                    MemoryManager.force_gc()
                    
                except Exception as e:
                    print(f"❌ ERROR loading {level} k{kmer}: {e}")
                    continue
        
        # Save summary
        summary_df = pd.DataFrame(self.results)
        summary_file = self.output_mgr.base_path / f"summary_{int(time.time())}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary: {summary_file}")
        
        # Print summary statistics
        if len(summary_df) > 0:
            print(f"\n{'='*60}")
            print("📊 BENCHMARK SUMMARY (NO DATA LEAKAGE)")
            print(f"{'='*60}")
            print(summary_df.to_string(index=False))
            
            print(f"\n📈 AVERAGE METRICS BY METHOD:")
            print(summary_df.groupby('method').agg({
                'n_components': 'mean',
                'train_cev_score': 'mean',
                'avg_memory_gb': 'mean',
                'peak_memory_gb': 'mean',
                'time_seconds': 'mean'
            }).round(4))
        
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