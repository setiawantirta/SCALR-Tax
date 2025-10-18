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
        
        # 🆕 SAMPLING PARAMETERS
        enable_sampling: bool = False,  # Enable stratified sampling
        sampling_kmer_threshold: int = 8,  # Apply sampling for k >= 8
        sampling_percentage: float = 0.1,  # 10% per class
        min_samples_per_class: int = 2,  # Minimum samples to keep class
        max_samples_per_class: int = None,  # Optional max limit

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

        # 🆕 Sampling config
        self.enable_sampling = enable_sampling
        self.sampling_kmer_threshold = sampling_kmer_threshold
        self.sampling_percentage = sampling_percentage
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_class = max_samples_per_class

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
# 3. STRATIFIED SAMPLING BARIS JIKA KMER > 8
# =================================================================
class StratifiedSampler:
    """
    Stratified sampling untuk large k-mer datasets
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def should_sample(self, kmer: int) -> bool:
        """Check if sampling should be applied"""
        return (
            self.config.enable_sampling and 
            kmer >= self.config.sampling_kmer_threshold
        )
    
    def sample_data(self, X, y, kmer: int, level: str, split: str = 'train'):
        """
        Perform stratified sampling per class
        
        Args:
            X: Sparse feature matrix
            y: Label array
            kmer: K-mer size
            level: Taxonomic level
            split: 'train' or 'test'
            
        Returns:
            X_sampled, y_sampled: Sampled data
            sampling_info: Dict with sampling statistics
        """
        
        if not self.should_sample(kmer):
            print(f"   ℹ️  No sampling (k-mer {kmer} < threshold {self.config.sampling_kmer_threshold})")
            return X, y, None
        
        print(f"\n{'='*60}")
        print(f"🎲 STRATIFIED SAMPLING ACTIVATED")
        print(f"{'='*60}")
        print(f"   K-mer: {kmer} >= {self.config.sampling_kmer_threshold}")
        print(f"   Level: {level}")
        print(f"   Split: {split}")
        print(f"   Original shape: {X.shape}")
        print(f"   Target sampling: {self.config.sampling_percentage*100:.1f}% per class")
        print(f"   Min samples per class: {self.config.min_samples_per_class}")
        
        # Get unique classes
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)
        
        print(f"\n📊 CLASS DISTRIBUTION:")
        print(f"   Total classes: {n_classes}")
        print(f"   Total samples: {len(y):,}")
        
        # Categorize classes
        small_classes = []  # < min_samples_per_class
        valid_classes = []  # >= min_samples_per_class
        
        for cls, count in zip(unique_classes, class_counts):
            if count < self.config.min_samples_per_class:
                small_classes.append((cls, count))
            else:
                valid_classes.append((cls, count))
        
        print(f"\n   Classes with >= {self.config.min_samples_per_class} samples: {len(valid_classes)}")
        print(f"   Classes with < {self.config.min_samples_per_class} samples: {len(small_classes)}")
        
        if small_classes:
            print(f"\n   ⚠️  Small classes (will be SKIPPED):")
            for cls, count in small_classes[:10]:  # Show first 10
                print(f"      Class {cls}: {count} samples")
            if len(small_classes) > 10:
                print(f"      ... and {len(small_classes)-10} more")
        
        # Perform stratified sampling
        sampled_indices = []
        sampling_stats = []
        skipped_classes = []
        
        print(f"\n🔄 Sampling per class...")
        
        for cls, original_count in tqdm(zip(unique_classes, class_counts), 
                                       total=n_classes, 
                                       desc="Sampling classes"):
            
            # Skip small classes
            if original_count < self.config.min_samples_per_class:
                skipped_classes.append(cls)
                continue
            
            # Get indices for this class
            class_mask = (y == cls)
            class_indices = np.where(class_mask)[0]
            
            # Calculate target sample size
            target_samples = max(
                self.config.min_samples_per_class,
                int(original_count * self.config.sampling_percentage)
            )
            
            # Apply max limit if specified
            if self.config.max_samples_per_class is not None:
                target_samples = min(target_samples, self.config.max_samples_per_class)
            
            # Don't sample more than available
            target_samples = min(target_samples, original_count)
            
            # Random sampling
            if target_samples < original_count:
                sampled_class_indices = np.random.choice(
                    class_indices, 
                    size=target_samples, 
                    replace=False
                )
            else:
                sampled_class_indices = class_indices
            
            sampled_indices.extend(sampled_class_indices)
            
            # Track statistics
            sampling_stats.append({
                'class': cls,
                'original': original_count,
                'sampled': len(sampled_class_indices),
                'percentage': (len(sampled_class_indices) / original_count) * 100
            })
        
        # Sort indices to maintain some order
        sampled_indices = np.sort(sampled_indices)
        
        # Extract sampled data
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        
        # Calculate statistics
        total_original = len(y)
        total_sampled = len(sampled_indices)
        reduction_percent = (1 - total_sampled / total_original) * 100
        
        sampling_info = {
            'original_samples': total_original,
            'sampled_samples': total_sampled,
            'reduction_percent': reduction_percent,
            'original_classes': n_classes,
            'valid_classes': len(valid_classes),
            'skipped_classes': len(skipped_classes),
            'per_class_stats': sampling_stats,
            'skipped_class_ids': skipped_classes
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ SAMPLING COMPLETED")
        print(f"{'='*60}")
        print(f"   Original: {total_original:,} samples, {n_classes} classes")
        print(f"   Sampled:  {total_sampled:,} samples, {len(valid_classes)} classes")
        print(f"   Reduction: {reduction_percent:.2f}%")
        print(f"   Skipped: {len(skipped_classes)} classes (< {self.config.min_samples_per_class} samples)")
        
        # Show per-class statistics (first 10)
        if sampling_stats:
            print(f"\n📊 PER-CLASS SAMPLING (showing first 10):")
            stats_df = pd.DataFrame(sampling_stats[:10])
            print(stats_df.to_string(index=False))
            
            if len(sampling_stats) > 10:
                print(f"   ... and {len(sampling_stats)-10} more classes")
            
            # Overall statistics
            avg_sampling = np.mean([s['percentage'] for s in sampling_stats])
            print(f"\n   Average sampling per class: {avg_sampling:.2f}%")
        
        MemoryManager.force_gc()
        
        return X_sampled, y_sampled, sampling_info
    
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
        🎨 Plot PCA/Feature Space 3D dengan SEMUA LABEL
        FIXED: Legend di kanan, colorbar untuk banyak label
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
            
            # ✅ LARGER FIGURE SIZE
            if n_classes <= 10:
                figsize = (18, 12)
            elif n_classes <= 30:
                figsize = (20, 14)
            else:
                figsize = (22, 16)
            
            fig = plt.figure(figsize=figsize)
            
            # ✅ ADJUST SUBPLOT POSITION - Lebih ke kiri untuk beri ruang legend/colorbar
            ax = fig.add_subplot(111, projection='3d', position=[0.05, 0.05, 0.65, 0.9])
            
            # Generate colors
            if n_classes <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, 20))
            else:
                colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            # ✅ LOGIC: Legend (<= 30 classes) vs Colorbar (> 30 classes)
            if n_classes <= 30:
                # ========================================
                # OPTION A: LEGEND DI KANAN (≤30 classes)
                # ========================================
                print(f"   🎨 Using LEGEND for {n_classes} classes (right side)")
                
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
                
                # ✅ LEGEND DI KANAN (BUKAN DI BAWAH!)
                # Adjust ncol based on number of classes
                if n_classes <= 15:
                    ncol = 1
                    fontsize = 8
                elif n_classes <= 25:
                    ncol = 2
                    fontsize = 7
                else:
                    ncol = 2
                    fontsize = 6
                
                ax.legend(
                    loc='upper left', 
                    bbox_to_anchor=(1.15, 1.0),  # Di kanan plot
                    fontsize=fontsize,
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    borderaxespad=0,
                    ncol=ncol  # Multiple columns if needed
                )
                
            else:
                # ========================================
                # OPTION B: COLORBAR DI KANAN (>30 classes)
                # ========================================
                print(f"   🎨 Using COLORBAR for {n_classes} classes (too many for legend)")
                
                from matplotlib.colors import Normalize
                from matplotlib.cm import ScalarMappable
                
                # Create color mapping
                norm = Normalize(vmin=unique_classes.min(), vmax=unique_classes.max())
                cmap = plt.cm.rainbow
                
                # Plot dengan colormap
                scatter = ax.scatter(
                    X_plot[:, 0], 
                    X_plot[:, 1],
                    X_plot[:, 2],
                    c=y_plot,  # Color by label
                    cmap=cmap,
                    norm=norm,
                    alpha=0.6,
                    s=20,
                    edgecolors='black',
                    linewidth=0.2
                )
                
                # ✅ ADD COLORBAR DI KANAN
                cbar = fig.colorbar(
                    scatter, 
                    ax=ax, 
                    pad=0.1,  # Distance from plot
                    shrink=0.8,  # Size of colorbar
                    aspect=20  # Aspect ratio
                )
                cbar.set_label('Class Label', fontsize=11, fontweight='bold', rotation=270, labelpad=20)
                cbar.ax.tick_params(labelsize=8)
            
            # ✅ STYLING dengan extra padding untuk axis labels
            ax.set_xlabel('PC1', fontsize=12, fontweight='bold', labelpad=15)
            ax.set_ylabel('PC2', fontsize=12, fontweight='bold', labelpad=15)
            ax.set_zlabel('PC3', fontsize=12, fontweight='bold', labelpad=15)
            
            # ✅ ADJUST AXIS LIMITS untuk prevent clipping
            z_min, z_max = X_plot[:, 2].min(), X_plot[:, 2].max()
            z_range = z_max - z_min
            ax.set_zlim(z_min - 0.1 * z_range, z_max + 0.1 * z_range)
            
            title = (f'3D PCA Feature Space - {level.upper()} K{kmer} ({method.upper()}) [{split.upper()}]\n'
                    f'{len(X_plot):,} samples | {n_classes} classes')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # ✅ ADJUST VIEW ANGLE untuk better visibility
            ax.view_init(elev=20, azim=45)  # Optimal viewing angle
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # ✅ TIGHT LAYOUT dengan extra padding
            plt.tight_layout(pad=2.0)
            
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
        self.sampler = StratifiedSampler(config)  # 🆕 Add sampler
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
        """Main benchmark loop - WITH STRATIFIED SAMPLING"""
        print("🚀 Starting Memory-Safe Benchmark with Stratified Sampling")
        print(f"⚙️  Max memory per operation: {self.config.max_memory_gb}GB")
        
        if self.config.enable_sampling:
            print(f"🎲 Sampling enabled for k-mer >= {self.config.sampling_kmer_threshold}")
            print(f"   Target: {self.config.sampling_percentage*100:.1f}% per class")
            print(f"   Min samples: {self.config.min_samples_per_class}")
        
        MemoryManager.log_memory("Initial")
        
        for level in self.config.levels:
            for kmer in self.config.kmers:
                print(f"\n{'='*60}")
                print(f"🧬 {level.upper()} - K-mer {kmer}")
                
                try:
                    # ✅ LOAD BOTH TRAIN & TEST
                    X_train_full, y_train_full = loader.load_data(level, kmer, 'train')
                    X_test_full, y_test_full = loader.load_data(level, kmer, 'test')
                    print(f"✅ Train data loaded: {X_train_full.shape}")
                    print(f"✅ Test data loaded: {X_test_full.shape}")
                    
                    # 🆕 APPLY SAMPLING IF ENABLED
                    X_train, y_train, train_sampling_info = self.sampler.sample_data(
                        X_train_full, y_train_full, kmer, level, split='train'
                    )
                    
                    X_test, y_test, test_sampling_info = self.sampler.sample_data(
                        X_test_full, y_test_full, kmer, level, split='test'
                    )
                    
                    # Clean up full data
                    del X_train_full, y_train_full, X_test_full, y_test_full
                    MemoryManager.force_gc()
                    
                    for method in self.config.methods:
                        print(f"\n🔧 Method: {method.upper()}")
                        
                        # Track memory usage
                        memory_samples = []
                        
                        try:
                            start = time.time()
                            mem_start = MemoryManager.get_memory_info()
                            memory_samples.append(mem_start['rss_gb'])
                            
                            # =================================================
                            # STEP 1: FIND OPTIMAL COMPONENTS (SAMPLED TRAIN)
                            # =================================================
                            print(f"\n📊 Step 1: Finding optimal components on {'SAMPLED ' if train_sampling_info else ''}TRAIN data...")
                            n_comp, variance_ratio, train_cev, fitted_model = self.find_optimal_components(
                                X_train, y_train, method
                            )
                            print(f"   ✅ Optimal components: {n_comp}")
                            print(f"   📈 Train CEV Score: {train_cev:.4f}")
                            
                            mem_after_opt = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_opt['rss_gb'])
                            
                            # =================================================
                            # STEP 2: TRANSFORM TRAIN DATA
                            # =================================================
                            print(f"\n🔄 Step 2: Transforming TRAIN data...")
                            X_train_transformed = self._transform_with_model(
                                X_train, fitted_model, method
                            )
                            
                            mem_after_train_transform = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_train_transform['rss_gb'])
                            
                            # =================================================
                            # STEP 3: TRANSFORM TEST DATA
                            # =================================================
                            print(f"\n🔄 Step 3: Transforming TEST data...")
                            X_test_transformed = self._transform_with_model(
                                X_test, fitted_model, method
                            )
                            
                            mem_after_test_transform = MemoryManager.get_memory_info()
                            memory_samples.append(mem_after_test_transform['rss_gb'])
                            
                            elapsed = time.time() - start
                            
                            # Calculate average memory usage
                            avg_memory_gb = np.mean(memory_samples)
                            peak_memory_gb = np.max(memory_samples)
                            
                            # =================================================
                            # STEP 4: SAVE RESULTS WITH SAMPLING INFO
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
                            
                            # 🆕 Save sampling info if available
                            if train_sampling_info:
                                output_dir = self.output_mgr.get_output_dir(level, kmer, method)
                                
                                # Save train sampling stats
                                sampling_df = pd.DataFrame(train_sampling_info['per_class_stats'])
                                sampling_file = output_dir / "sampling_stats_train.csv"
                                sampling_df.to_csv(sampling_file, index=False)
                                print(f"   💾 Train sampling stats: {sampling_file}")
                                
                                # Save summary
                                summary_dict = {
                                    'original_samples': train_sampling_info['original_samples'],
                                    'sampled_samples': train_sampling_info['sampled_samples'],
                                    'reduction_percent': train_sampling_info['reduction_percent'],
                                    'original_classes': train_sampling_info['original_classes'],
                                    'valid_classes': train_sampling_info['valid_classes'],
                                    'skipped_classes': train_sampling_info['skipped_classes']
                                }
                                
                                summary_file = output_dir / "sampling_summary_train.txt"
                                with open(summary_file, 'w') as f:
                                    for key, value in summary_dict.items():
                                        f.write(f"{key}: {value}\n")
                                print(f"   💾 Train sampling summary: {summary_file}")
                            
                            if test_sampling_info:
                                output_dir = self.output_mgr.get_output_dir(level, kmer, method)
                                
                                # Save test sampling stats
                                sampling_df = pd.DataFrame(test_sampling_info['per_class_stats'])
                                sampling_file = output_dir / "sampling_stats_test.csv"
                                sampling_df.to_csv(sampling_file, index=False)
                                print(f"   💾 Test sampling stats: {sampling_file}")

                            # =================================================
                            # STEP 5: CREATE PLOTS
                            # =================================================
                            if self.plotter:
                                output_dir = self.output_mgr.get_output_dir(level, kmer, method)
                                print(f"\n📊 Step 5: Creating enhanced plots...")
                                
                                # Variance Explained
                                if variance_ratio is not None:
                                    self.plotter.plot_variance_explained(
                                        variance_ratio, n_comp, level, kmer, method, output_dir
                                    )
                                
                                # PCA Feature Space - TRAIN
                                self.plotter.plot_pca_feature_space_all_labels(
                                    X_train_transformed, y_train, level, kmer, 
                                    method, output_dir, split='train'
                                )
                                
                                # PCA Feature Space - TEST
                                self.plotter.plot_pca_feature_space_all_labels(
                                    X_test_transformed, y_test, level, kmer, 
                                    method, output_dir, split='test'
                                )
                                
                                # 3D Feature Space - TRAIN
                                if X_train_transformed.shape[1] >= 3:
                                    self.plotter.plot_pca_feature_space_3d(
                                        X_train_transformed, y_train, level, kmer, 
                                        method, output_dir, split='train'
                                    )
                                
                                # Class Separation Heatmap
                                self.plotter.plot_class_separation_heatmap(
                                    X_train_transformed, y_train, level, kmer, 
                                    method, output_dir, split='train'
                                )
                                
                                # Component Distribution
                                self.plotter.plot_component_distribution(
                                    X_train_transformed, level, kmer, method, output_dir
                                )

                            # =================================================
                            # STEP 6: SAVE METRICS
                            # =================================================
                            result_dict = {
                                'level': level,
                                'kmer': kmer,
                                'method': method,
                                'n_components': n_comp,
                                'train_cev_score': train_cev,
                                'avg_memory_gb': avg_memory_gb,
                                'peak_memory_gb': peak_memory_gb,
                                'time_seconds': elapsed
                            }
                            
                            # 🆕 Add sampling info to results
                            if train_sampling_info:
                                result_dict.update({
                                    'sampling_enabled': True,
                                    'original_train_samples': train_sampling_info['original_samples'],
                                    'sampled_train_samples': train_sampling_info['sampled_samples'],
                                    'train_reduction_percent': train_sampling_info['reduction_percent'],
                                    'original_classes': train_sampling_info['original_classes'],
                                    'valid_classes': train_sampling_info['valid_classes'],
                                    'skipped_classes': train_sampling_info['skipped_classes']
                                })
                            else:
                                result_dict['sampling_enabled'] = False
                            
                            self.results.append(result_dict)
                            
                            print(f"\n✅ SUCCESS: {method} completed!")
                            print(f"   📊 Train CEV Score: {train_cev:.4f}")
                            if train_sampling_info:
                                print(f"   🎲 Sampled: {train_sampling_info['sampled_samples']:,} / "
                                      f"{train_sampling_info['original_samples']:,} samples "
                                      f"({train_sampling_info['reduction_percent']:.1f}% reduction)")
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
            print("📊 BENCHMARK SUMMARY")
            print(f"{'='*60}")
            print(summary_df.to_string(index=False))
        
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

    # 🆕 SAMPLING PARAMETERS
    enable_sampling: bool = False,
    sampling_kmer_threshold: int = 8,
    sampling_percentage: float = 0.1,
    min_samples_per_class: int = 2,
    max_samples_per_class: int = None,
    
    **kwargs
):
    """
    🚀 MEMORY-SAFE BENCHMARK API with Stratified Sampling
    
    Args:
        data_path: Path ke vectorization directory
        levels: List taxonomic levels
        kmers: List k-mer sizes
        methods: List methods ('ipca', 'svd')
        output_dir: Output directory
        create_plots: Whether to create plots
        max_memory_gb: Maximum memory per operation (GB)
        
        # 🆕 Sampling parameters
        enable_sampling: Enable stratified sampling
        sampling_kmer_threshold: Apply sampling for k-mer >= threshold
        sampling_percentage: Target percentage per class (0.0-1.0)
        min_samples_per_class: Minimum samples to keep class
        max_samples_per_class: Maximum samples per class (optional)
        
    Returns:
        DataFrame dengan hasil benchmark
        
    Example 1: Regular (no sampling)
        results = run_benchmark(
            data_path='/path/to/data',
            levels=['genus'],
            kmers=[6],
            methods=['ipca']
        )
    
    Example 2: With sampling for large k-mer
        results = run_benchmark(
            data_path='/path/to/data',
            levels=['genus'],
            kmers=[6, 8, 10],
            methods=['ipca'],
            enable_sampling=True,
            sampling_kmer_threshold=8,  # Sample k>=8 only
            sampling_percentage=0.1,     # 10% per class
            min_samples_per_class=2,     # Skip classes with <2 samples
            max_samples_per_class=1000   # Max 1000 per class
        )
    """

    config = BenchmarkConfig(
        levels=levels,
        kmers=kmers,
        methods=methods,
        output_dir=output_dir,
        create_plots=create_plots,
        max_memory_gb=max_memory_gb,

        # 🆕 Sampling config
        enable_sampling=enable_sampling,
        sampling_kmer_threshold=sampling_kmer_threshold,
        sampling_percentage=sampling_percentage,
        min_samples_per_class=min_samples_per_class,
        max_samples_per_class=max_samples_per_class,
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
    print("STRATIFIED SAMPLING - USAGE EXAMPLES")
    print("=" * 80)
    
    # Example 1: Regular k-mer (no sampling)
    print("\n📝 Example 1: K-mer 6 (no sampling)")
    print("""
    results = run_benchmark(
        data_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization',
        levels=['genus'],
        kmers=[6],
        methods=['ipca'],
        enable_sampling=False  # Disabled
    )
    """)
    
    # Example 2: Large k-mer with sampling
    print("\n📝 Example 2: K-mer 8+ (with 10% sampling)")
    print("""
    results = run_benchmark(
        data_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization',
        levels=['genus'],
        kmers=[8, 10],
        methods=['ipca'],
        enable_sampling=True,
        sampling_kmer_threshold=8,   # Sample k>=8
        sampling_percentage=0.1,      # 10% per class
        min_samples_per_class=2,      # Skip if <2 samples
        max_memory_gb=4.0
    )
    """)
    
    # Example 3: Conservative sampling
    print("\n📝 Example 3: Conservative sampling (20% per class)")
    print("""
    results = run_benchmark(
        data_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization',
        levels=['genus'],
        kmers=[6, 8],
        methods=['ipca'],
        enable_sampling=True,
        sampling_kmer_threshold=8,
        sampling_percentage=0.2,      # 20% per class
        min_samples_per_class=5,      # Skip if <5 samples
        max_samples_per_class=5000    # Max 5000 per class
    )
    """)
    
    print("\n✅ Key sampling features:")
    print("   ✓ Stratified sampling per class")
    print("   ✓ Configurable percentage (default 10%)")
    print("   ✓ Auto-skip small classes (<min_samples)")
    print("   ✓ Optional max limit per class")
    print("   ✓ Detailed sampling statistics saved")
    print("   ✓ Applied only to k-mer >= threshold")
    
    print("\n📊 Output files:")
    print("   - features_fold0_train.csv (sampled features)")
    print("   - sampling_stats_train.csv (per-class details)")
    print("   - sampling_summary_train.txt (overall summary)")