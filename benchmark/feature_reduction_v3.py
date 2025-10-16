
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import psutil
import gc
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import warnings
import itertools
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pickle
import json
import seaborn as sns
from itertools import product
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from scipy import sparse
import os
import pickle
import glob
from pathlib import Path
import re
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Standard imports
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.utils.extmath import randomized_svd

class OutputDirectoryManager:
    """
    Manages dynamic output directory structure for benchmark results.
    
    This class handles:
    - Creating directory structures based on level/kmer/method/device combinations
    - Managing file skip/overwrite options
    - Tracking all output file paths in a text file
    
    Directory structure: {base_path}/{level}/{kmer}/{method}/{device}/
    
    Attributes:
        base_output_path (Path): Base directory for all outputs
        skip_existing (bool): If True, skip if file exists; if False, overwrite
        output_paths_file (Path): File to track all output paths
        created_paths (set): Set of already created directory paths
        saved_files (list): List of all saved file paths
    
    Example:
        manager = OutputDirectoryManager(
            base_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction',
            skip_existing=True
        )
        
        # Get output path for specific combination
        output_dir = manager.get_output_directory('genus', 6, 'ipca', 'cpu')
        
        # Save file with automatic path tracking
        file_path = manager.save_file(
            content=data, 
            filename='features.csv',
            level='genus', kmer=6, method='ipca', device='cpu'
        )
    """
    
    def __init__(self, base_path: str = '/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction', 
                 skip_existing: bool = True):
        """
        Initialize output directory manager.
        
        Args:
            base_path: Base directory path for all outputs
            skip_existing: If True, skip existing files; if False, overwrite
        """
        self.base_output_path = Path(base_path)
        self.skip_existing = skip_existing
        self.output_paths_file = self.base_output_path / 'output_paths_log.txt'
        self.created_paths = set()
        self.saved_files = []
        
        # Create base directory
        self.base_output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🗂️  Output Directory Manager initialized:")
        print(f"   - Base path: {self.base_output_path}")
        print(f"   - Skip existing: {self.skip_existing}")
        print(f"   - Paths log: {self.output_paths_file}")
    
    def get_output_directory(self, level: str, kmer: int, method: str, device: str) -> Path:
        """
        Get output directory path for specific combination.
        
        Args:
            level: Taxonomic level (e.g., 'genus', 'species')
            kmer: K-mer size (e.g., 6, 8, 10)
            method: Method name (e.g., 'ipca', 'svd', 'umap', 'autoencoder')
            device: Device type (e.g., 'cpu', 'gpu')
            
        Returns:
            Path object for the specific combination directory
        """
        output_dir = self.base_output_path / level / f'k{kmer}' / method / device
        
        # Create directory if it doesn't exist
        if output_dir not in self.created_paths:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.created_paths.add(output_dir)
            print(f"📁 Created directory: {output_dir}")
            
        return output_dir
    
    def should_skip_file(self, file_path: Path) -> bool:
        """
        Check if file should be skipped based on skip_existing setting.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file should be skipped, False if should proceed
        """
        if not self.skip_existing:
            return False
            
        if file_path.exists():
            print(f"⏭️  Skipping existing file: {file_path}")
            return True
            
        return False
    
    def log_output_path(self, file_path: Path, level: str, kmer: int, method: str, device: str, file_type: str = 'unknown'):
        """
        Log output path to the tracking file.
        
        Args:
            file_path: Path of the saved file
            level: Taxonomic level
            kmer: K-mer size
            method: Method name
            device: Device type
            file_type: Type of file (e.g., 'features', 'plot', 'cev', 'manifold')
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp}\t{level}\tk{kmer}\t{method}\t{device}\t{file_type}\t{file_path}\n"
        
        with open(self.output_paths_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            
        self.saved_files.append({
            'timestamp': timestamp,
            'level': level,
            'kmer': kmer,
            'method': method,
            'device': device,
            'file_type': file_type,
            'path': str(file_path)
        })
    
    def save_file_with_tracking(self, content, filename: str, level: str, kmer: int, 
                               method: str, device: str, file_type: str = 'data',
                               save_func=None) -> Optional[Path]:
        """
        Save file with automatic path tracking and skip/overwrite handling.
        
        Args:
            content: Content to save (DataFrame, numpy array, etc.)
            filename: Name of the file to save
            level: Taxonomic level
            kmer: K-mer size
            method: Method name
            device: Device type
            file_type: Type of file for logging
            save_func: Custom save function (if None, will try to auto-detect)
            
        Returns:
            Path of saved file, or None if skipped
        """
        output_dir = self.get_output_directory(level, kmer, method, device)
        file_path = output_dir / filename
        
        # Check if should skip
        if self.should_skip_file(file_path):
            return None
            
        try:
            # Auto-detect save function based on content type and file extension
            if save_func:
                save_func(content, file_path)
            elif hasattr(content, 'to_csv') and filename.endswith('.csv'):
                content.to_csv(file_path, index=False)
            elif hasattr(content, 'savefig'):  # matplotlib figure
                content.savefig(file_path, dpi=300, bbox_inches='tight')
            elif isinstance(content, (np.ndarray, list)):
                if filename.endswith('.npy'):
                    np.save(file_path, content)
                elif filename.endswith('.csv'):
                    np.savetxt(file_path, content, delimiter=',')
                else:
                    # Default to pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(content, f)
            elif isinstance(content, str):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Default to pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(content, f)
                    
            # Log the saved file
            self.log_output_path(file_path, level, kmer, method, device, file_type)
            print(f"💾 Saved {file_type}: {file_path}")
            
            return file_path
            
        except Exception as e:
            print(f"❌ Failed to save {file_type} to {file_path}: {str(e)}")
            return None
    
    def get_summary_report(self) -> str:
        """
        Generate summary report of all saved files.
        
        Returns:
            String containing summary report
        """
        if not self.saved_files:
            return "No files saved yet."
            
        report = f"📊 OUTPUT SUMMARY REPORT\n"
        report += f"{'='*80}\n"
        report += f"Total files saved: {len(self.saved_files)}\n"
        report += f"Base output path: {self.base_output_path}\n\n"
        
        # Group by combination
        from collections import defaultdict
        by_combination = defaultdict(list)
        
        for file_info in self.saved_files:
            key = f"{file_info['level']}-k{file_info['kmer']}-{file_info['method']}-{file_info['device']}"
            by_combination[key].append(file_info)
        
        report += f"Files by combination:\n"
        for combo, files in by_combination.items():
            report += f"  {combo}: {len(files)} files\n"
            for file_info in files:
                report += f"    - {file_info['file_type']}: {Path(file_info['path']).name}\n"
        
        return report
    
    def save_summary_report(self) -> Path:
        """
        Save summary report to file.
        
        Returns:
            Path to the summary report file
        """
        report_content = self.get_summary_report()
        report_path = self.base_output_path / f'summary_report_{int(time.time())}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"📋 Summary report saved: {report_path}")
        return report_path

def safe_to_dense(X):
    """Safely convert sparse matrix to dense array"""
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X

def safe_slice_to_dense(X, start_idx, end_idx):
    """Safely slice and convert to dense array"""
    if hasattr(X, 'toarray'):
        return X[start_idx:end_idx].toarray()
    return X[start_idx:end_idx]

# UMAP import
try:
    import umap
    UMAP_AVAILABLE = True
    print("✅ UMAP tersedia")
except ImportError:
    UMAP_AVAILABLE = False
    print("❌ UMAP tidak tersedia")

# PyTorch imports for Autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("✅ PyTorch tersedia")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch tidak tersedia")

# GPU imports (optional)
# GPU imports (optional) - FIXED VERSION
import os

# 🔧 FIX: Disable GPU before any cuML imports
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CUML_AVAILABLE = False  # Default to False

# Only try to import cuML if explicitly needed and CUDA is available
if os.environ.get('ENABLE_GPU_BENCHMARK', '0') == '1':
    try:
        import cupy as cp
        # Test CUDA availability
        cp.cuda.runtime.getDeviceCount()
        
        # Only import cuML if CUDA actually works
        from cuml.decomposition import TruncatedSVD as CuTruncatedSVD
        from cuml.decomposition import IncrementalPCA as CuIncrementalPCA
        
        CUML_AVAILABLE = True
        print("✅ GPU (CuML) tersedia")
    except Exception as cuda_error:
        CUML_AVAILABLE = False
        print(f"⚠️ CuML tidak tersedia: {str(cuda_error)}")
        print("💡 Falling back to CPU-only mode")
else:
    print("🖥️  GPU disabled by environment variable")
    print("💡 Set ENABLE_GPU_BENCHMARK=1 to enable GPU support")
    
# try:
#     from cuml.decomposition import TruncatedSVD as CuTruncatedSVD
#     from cuml.decomposition import IncrementalPCA as CuIncrementalPCA
#     import cupy as cp
#     try:
#         import cuml
#         # Test CUDA compatibility
#         cp.cuda.runtime.getDeviceCount()
#         CUML_AVAILABLE = True
#         print("✅ GPU (CuML) tersedia")
#     except Exception as cuda_error:
#         CUML_AVAILABLE = False
#         print(f"❌ CuML tidak tersedia: {str(cuda_error)}")
#         print("💡 Falling back to CPU-only mode")
# except ImportError:
#     CUML_AVAILABLE = False
#     print("❌ CuML tidak tersedia, hanya menggunakan CPU")

# PyTorch GPU check yang lebih robust
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True

    # Test CUDA availability safely
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            test_tensor = torch.tensor([1.0]).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ PyTorch GPU tersedia")
        except Exception as cuda_error:
            print(f"⚠️ PyTorch CUDA error: {str(cuda_error)}")
            print("💡 PyTorch will use CPU only")
    else:
        print("✅ PyTorch tersedia (CPU only)")

except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch tidak tersedia")

@dataclass
class BenchmarkResult:
    """Enhanced result class with proper field definitions"""
    method: str = ""
    device: str = ""
    fold: int = 0
    optimal_n: int = 0
    max_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    total_time_seconds: float = 0.0
    search_time_seconds: float = 0.0
    transform_time_seconds: float = 0.0
    final_cev: float = 0.0
    silhouette_score: float = 0.0
    success: bool = False
    parameters: Dict = field(default_factory=dict)
    cev_csv_path: Optional[str] = None
    cev_plot_path: Optional[str] = None
    features_csv_path: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(time.time()))

    # Enhanced manifold learning fields
    trustworthiness_score: float = 0.0
    continuity_score: float = 0.0
    procrustes_score: float = 0.0
    local_preservation_score: float = 0.0
    intrinsic_dimension: float = 0.0
    manifold_combined_score: float = 0.0
    optimization_method: str = "cev"  # "cev" or "manifold"

    def to_dict(self):
        """Convert to dictionary with all fields"""
        return {
            'method': self.method,
            'device': self.device,
            'fold': self.fold,
            'optimal_n': self.optimal_n,
            'max_memory_mb': self.max_memory_mb,
            'avg_memory_mb': self.avg_memory_mb,
            'total_time_seconds': self.total_time_seconds,
            'search_time_seconds': self.search_time_seconds,
            'transform_time_seconds': self.transform_time_seconds,
            'final_cev': self.final_cev,
            'silhouette_score': self.silhouette_score,
            'success': self.success,
            'parameters': str(self.parameters),
            'cev_csv_path': self.cev_csv_path,
            'cev_plot_path': self.cev_plot_path,
            'features_csv_path': self.features_csv_path,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'trustworthiness_score': self.trustworthiness_score,
            'continuity_score': self.continuity_score,
            'procrustes_score': self.procrustes_score,
            'local_preservation_score': self.local_preservation_score,
            'intrinsic_dimension': self.intrinsic_dimension,
            'manifold_combined_score': self.manifold_combined_score,
            'optimization_method': self.optimization_method
        }

class MemoryMonitor:
    """Enhanced class untuk monitoring penggunaan memory real-time"""

    def __init__(self):
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.monitoring = False
        self.monitor_thread = None
        self.baseline_memory = psutil.virtual_memory().used / 1024 / 1024
        self.baseline_gpu_memory = 0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.baseline_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

    def start_monitoring(self):
        """Mulai monitoring memory"""
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.baseline_memory = psutil.virtual_memory().used / 1024 / 1024

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.baseline_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring memory"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_memory(self):
        """Internal method untuk monitoring memory"""
        while self.monitoring:
            # CPU Memory
            current_memory = psutil.virtual_memory().used / 1024 / 1024
            memory_delta = current_memory - self.baseline_memory
            self.memory_usage.append(max(0, memory_delta))

            # GPU Memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_delta = current_gpu_memory - self.baseline_gpu_memory
                self.gpu_memory_usage.append(max(0, gpu_memory_delta))

            time.sleep(0.1)

    def get_stats(self) -> Tuple[float, float, float, float]:
        """Get max dan average memory usage for both CPU and GPU"""
        if not self.memory_usage:
            return 0.0, 0.0, 0.0, 0.0

        cpu_max = max(self.memory_usage)
        cpu_avg = np.mean(self.memory_usage)

        if self.gpu_memory_usage:
            gpu_max = max(self.gpu_memory_usage)
            gpu_avg = np.mean(self.gpu_memory_usage)
        else:
            gpu_max, gpu_avg = 0.0, 0.0

        return cpu_max, cpu_avg, gpu_max, gpu_avg

class SimpleAutoencoder(nn.Module):
    """Simple Autoencoder for dimensionality reduction"""

    def __init__(self, input_dim, encoding_dim):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, encoding_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        return self.encoder(x)

class EnhancedDimensionalityReductionBenchmark:
    """Enhanced Main class untuk comprehensive benchmark dimensionality reduction"""

    def __init__(self, cev_threshold=0.95, start_components=100, step_components=50,
                 max_limit=1000, batch_size=500, cv_folds=3, autoencoder_epochs=50,
                 umap_n_neighbors=15, taxonomic_levels=['genus', 'species', 'family', 'order', 'phylum'],
                 kmer_sizes=[6, 8, 10], enable_manifold_metrics=True,
                 output_directory=None, skip_existing=True):

        self.cev_threshold = cev_threshold
        self.start_components = start_components
        self.step_components = step_components
        self.max_limit = max_limit
        self.batch_size = batch_size
        self.enable_manifold_metrics = enable_manifold_metrics  # Fixed: Added missing parameter
        self.cv_folds = max(1, cv_folds)  # Ensure at least 1 fold
        self.autoencoder_epochs = autoencoder_epochs
        self.umap_n_neighbors = umap_n_neighbors
        self.results = []
        self.timestamp = int(time.time())
        self.all_features = {}  # Store extracted features

        self.taxonomic_levels = taxonomic_levels
        self.kmer_sizes = kmer_sizes
        self.grid_search_results = []
        self.all_results_df = None
        self.production_features = {}  # Store features for each combination

        # Initialize output directory manager
        if output_directory is None:
            output_directory = '/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction'
        
        self.output_manager = OutputDirectoryManager(
            base_path=output_directory,
            skip_existing=skip_existing
        )

        # Validation
        if self.batch_size <= self.start_components:
            self.batch_size = self.start_components + 100
            print(f"⚠️  Auto-adjusted batch_size to {self.batch_size}")

        print(f"🔧 ENHANCED PARAMETERS:")
        print(f"   - CEV Threshold: {self.cev_threshold}")
        print(f"   - Components range: {self.start_components} to {self.max_limit} (step: {self.step_components})")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - CV Folds: {self.cv_folds}")
        print(f"   - Autoencoder epochs: {self.autoencoder_epochs}")
        print(f"   - UMAP neighbors: {self.umap_n_neighbors}")
        print(f"   - Output directory: {self.output_manager.base_output_path}")
        print(f"   - Skip existing: {self.output_manager.skip_existing}")

        print(f"🏭 PRODUCTION MODE INITIALIZED:")
        print(f"   - Taxonomic levels: {self.taxonomic_levels}")
        print(f"   - K-mer sizes: {self.kmer_sizes}")
        print(f"   - Total combinations: {len(self.taxonomic_levels) * len(self.kmer_sizes)}")

    def create_stratified_sample(self, X_sparse, y, sample_size, random_state=42):
        """Create stratified sample for SVD"""
        from sklearn.model_selection import train_test_split

        if len(np.unique(y)) == 1:
            # Single class, use random sampling
            np.random.seed(random_state)
            indices = np.random.choice(len(y), sample_size, replace=False)
            return X_sparse[indices], y[indices], indices

        try:
            # Use stratified sampling
            _, _, _, _, indices_train, _ = train_test_split(
                range(len(y)), y,
                train_size=sample_size/len(y),
                stratify=y,
                random_state=random_state
            )
            return X_sparse[indices_train], y[indices_train], indices_train
        except:
            # Fallback to random sampling
            np.random.seed(random_state)
            indices = np.random.choice(len(y), sample_size, replace=False)
            return X_sparse[indices], y[indices], indices

    def adaptive_batch_size(self, n_components, data_shape):
        """Enhanced adaptive batch size with proper handling for small datasets"""
        n_samples = data_shape[0]
        
        # For very small datasets, use entire dataset as batch
        if n_samples <= 50:
            print(f"🔧 Small dataset ({n_samples} samples): using full batch")
            return n_samples
        
        # Calculate safe minimums for IPCA
        safe_min_batch = max(n_components + 10, min(20, n_samples // 2))
        max_batch_size = min(n_samples, 2000)
        
        # Ensure we don't exceed dataset size
        optimal_batch_size = min(max(safe_min_batch, self.batch_size), max_batch_size)
        
        # Final safety check
        optimal_batch_size = min(optimal_batch_size, n_samples)

        if optimal_batch_size != self.batch_size:
            print(f"🔧 Adaptive batch size: {self.batch_size} → {optimal_batch_size}")

        return optimal_batch_size

    def check_memory_usage(self, operation_name=""):
        """Enhanced memory check with GPU"""
        cpu_memory_percent = psutil.virtual_memory().percent

        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

            if cpu_memory_percent > 85 or gpu_memory_percent > 85:
                print(f"⚠️  HIGH MEMORY WARNING ({operation_name}): CPU {cpu_memory_percent:.1f}%, GPU {gpu_memory_percent:.1f}%")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            if cpu_memory_percent > 85:
                print(f"⚠️  HIGH MEMORY WARNING ({operation_name}): CPU {cpu_memory_percent:.1f}%")
                gc.collect()

    def check_gpu_memory(self):
        """Check and manage GPU memory"""
        if CUML_AVAILABLE:
            try:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                if total_bytes > 0 and used_bytes / total_bytes > 0.8:
                    cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            if torch.cuda.memory_allocated() > torch.cuda.get_device_properties(0).total_memory * 0.8:
                torch.cuda.empty_cache()

    def calculate_silhouette_score(self, X_transformed, y, sample_size=1000):
        """Calculate silhouette score with sampling for large datasets"""
        try:
            if len(np.unique(y)) < 2:
                return 0.0

            if X_transformed.shape[0] > sample_size:
                indices = np.random.choice(X_transformed.shape[0], sample_size, replace=False)
                X_sample = X_transformed[indices]
                y_sample = y[indices]
            else:
                X_sample = X_transformed
                y_sample = y

            return silhouette_score(X_sample, y_sample)
        except Exception as e:
            print(f"⚠️  Silhouette score calculation failed: {str(e)}")
            return 0.0


    def calculate_trustworthiness_continuity(self, X_original, X_embedded, k=12):
        """
        Calculate trustworthiness and continuity for manifold learning evaluation
        """
        try:
            from sklearn.manifold import trustworthiness
            from sklearn.neighbors import NearestNeighbors

            # Sample for large datasets to avoid memory issues
            sample_size = min(1000, len(X_original))
            if len(X_original) > sample_size:
                indices = np.random.choice(len(X_original), sample_size, replace=False)
                X_orig_sample = X_original[indices]
                X_emb_sample = X_embedded[indices]
            else:
                X_orig_sample = X_original
                X_emb_sample = X_embedded

            # Calculate trustworthiness
            trust = trustworthiness(X_orig_sample, X_emb_sample, n_neighbors=k)

            # Calculate continuity manually
            k_actual = min(k, len(X_orig_sample) - 1)

            # Original space neighbors
            nbrs_orig = NearestNeighbors(n_neighbors=k_actual+1).fit(X_orig_sample)
            _, indices_orig = nbrs_orig.kneighbors(X_orig_sample)

            # Embedded space neighbors
            nbrs_emb = NearestNeighbors(n_neighbors=k_actual+1).fit(X_emb_sample)
            _, indices_emb = nbrs_emb.kneighbors(X_emb_sample)

            continuity = 0
            for i in range(len(X_orig_sample)):
                orig_neighbors = set(indices_orig[i][1:])  # exclude self
                emb_neighbors = set(indices_emb[i][1:])
                continuity += len(orig_neighbors.intersection(emb_neighbors)) / k_actual

            continuity /= len(X_orig_sample)

            del X_orig_sample, X_emb_sample, nbrs_orig, nbrs_emb
            gc.collect()

            return trust, continuity

        except Exception as e:
            print(f"⚠️  Trustworthiness/Continuity calculation failed: {str(e)}")
            return 0.0, 0.0

    def calculate_procrustes_score(self, X_original, X_embedded, sample_size=1000):
        """
        Calculate procrustes analysis score for distance preservation
        """
        try:
            from scipy.spatial.distance import pdist

            if len(X_original) > sample_size:
                indices = np.random.choice(len(X_original), sample_size, replace=False)
                X_orig_sample = X_original[indices]
                X_emb_sample = X_embedded[indices]
            else:
                X_orig_sample = X_original
                X_emb_sample = X_embedded

            # Calculate pairwise distances
            dist_orig = pdist(X_orig_sample)
            dist_emb = pdist(X_emb_sample)

            # Normalize distances
            if np.max(dist_orig) > 0:
                dist_orig = dist_orig / np.max(dist_orig)
            if np.max(dist_emb) > 0:
                dist_emb = dist_emb / np.max(dist_emb)

            # Calculate correlation
            correlation = np.corrcoef(dist_orig, dist_emb)[0, 1]

            del X_orig_sample, X_emb_sample, dist_orig, dist_emb
            gc.collect()

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"⚠️  Procrustes score calculation failed: {str(e)}")
            return 0.0

    def calculate_local_neighborhood_preservation(self, X_original, X_embedded, k=15):
        """
        Calculate local neighborhood preservation using Jaccard similarity
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            sample_size = min(1000, len(X_original))
            if len(X_original) > sample_size:
                indices = np.random.choice(len(X_original), sample_size, replace=False)
                X_orig_sample = X_original[indices]
                X_emb_sample = X_embedded[indices]
            else:
                X_orig_sample = X_original
                X_emb_sample = X_embedded

            k_actual = min(k, len(X_orig_sample) - 1)

            # Find k-nearest neighbors in original space
            nbrs_orig = NearestNeighbors(n_neighbors=k_actual+1).fit(X_orig_sample)
            _, indices_orig = nbrs_orig.kneighbors(X_orig_sample)

            # Find k-nearest neighbors in embedded space
            nbrs_emb = NearestNeighbors(n_neighbors=k_actual+1).fit(X_emb_sample)
            _, indices_emb = nbrs_emb.kneighbors(X_emb_sample)

            preservation_scores = []
            for i in range(len(X_orig_sample)):
                orig_neighbors = set(indices_orig[i][1:])  # exclude self
                emb_neighbors = set(indices_emb[i][1:])

                # Jaccard similarity
                intersection = len(orig_neighbors.intersection(emb_neighbors))
                union = len(orig_neighbors.union(emb_neighbors))
                jaccard = intersection / union if union > 0 else 0

                preservation_scores.append(jaccard)

            result = np.mean(preservation_scores)

            del X_orig_sample, X_emb_sample, nbrs_orig, nbrs_emb
            gc.collect()

            return result

        except Exception as e:
            print(f"⚠️  Local neighborhood preservation calculation failed: {str(e)}")
            return 0.0

    def estimate_intrinsic_dimension(self, X_embedded):
        """
        Estimate intrinsic dimensionality using MLE method
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            sample_size = min(500, len(X_embedded))
            if len(X_embedded) > sample_size:
                indices = np.random.choice(len(X_embedded), sample_size, replace=False)
                X_sample = X_embedded[indices]
            else:
                X_sample = X_embedded

            k = min(20, len(X_sample) // 5)
            if k < 2:
                return 2

            nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)

            # Remove self-distances (first column)
            distances = distances[:, 1:]

            # MLE estimation
            intrinsic_dims = []
            for i in range(len(distances)):
                dists = distances[i]
                dists = dists[dists > 1e-10]  # Remove zero/tiny distances

                if len(dists) >= 2:
                    # MLE formula
                    r_k = dists[-1]  # furthest neighbor
                    if r_k > 0:
                        sum_log = np.sum(np.log(dists[:-1] / r_k))
                        if sum_log < 0:
                            intrinsic_dim = -(k-1) / sum_log
                            intrinsic_dims.append(max(1, min(50, intrinsic_dim)))

            result = np.median(intrinsic_dims) if intrinsic_dims else 2

            del X_sample, distances, nbrs
            gc.collect()

            return result

        except Exception as e:
            print(f"⚠️  Intrinsic dimension estimation failed: {str(e)}")
            return 2

    def save_manifold_optimization_results(self, results, method, device, fold, level=None, kmer_size=None):
        """
        Save manifold learning optimization results to CSV using output manager
        """
        try:
            if level is None or kmer_size is None:
                print("⚠️  Level and kmer_size required for organized output")
                return None
                
            filename = f"manifold_optimization_fold{fold}_{self.timestamp}.csv"

            df = pd.DataFrame(results)

            metadata = [
                f"# Manifold Learning Optimization Results",
                f"# Method: {method.upper()}",
                f"# Device: {device.upper()}",
                f"# Fold: {fold}",
                f"# Level: {level}",
                f"# K-mer Size: {kmer_size}",
                f"# Metrics Used: Trustworthiness, Continuity, Procrustes, Local Preservation",
                f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"#"
            ]

            # Prepare content with metadata
            content_lines = metadata + ['']  # Add empty line after metadata
            content_lines.append(df.to_csv(index=False))
            content = '\n'.join(content_lines)

            # Save using output manager
            file_path = self.output_manager.save_file_with_tracking(
                content=content,
                filename=filename,
                level=level,
                kmer=kmer_size,
                method=method,
                device=device,
                file_type='manifold_optimization',
                save_func=lambda c, p: p.write_text(c, encoding='utf-8')
            )

            return str(file_path) if file_path else None

        except Exception as e:
            print(f"⚠️  Failed to save manifold results: {str(e)}")
            return None

    def plot_manifold_optimization(self, results, method, device, fold, best_components, level=None, kmer_size=None):
        """
        Create optimization plot for manifold learning methods using output manager
        """
        try:
            if level is None or kmer_size is None:
                print("⚠️  Level and kmer_size required for organized output")
                return None
                
            df = pd.DataFrame(results)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Plot 1: Trustworthiness & Continuity
            ax1.plot(df['n_components'], df['trustworthiness'], 'o-', label='Trustworthiness', linewidth=2, markersize=6)
            ax1.plot(df['n_components'], df['continuity'], 's-', label='Continuity', linewidth=2, markersize=6)
            ax1.axvline(x=best_components, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_components}')
            ax1.set_xlabel('Number of Components')
            ax1.set_ylabel('Score')
            ax1.set_title('Trustworthiness & Continuity')
            ax1.legend()
            # ax1.grid(True, alpha=0.3)

            # Plot 2: Procrustes & Local Preservation
            ax2.plot(df['n_components'], df['procrustes'], '^-', label='Procrustes', linewidth=2, markersize=6)
            ax2.plot(df['n_components'], df['local_preservation'], 'v-', label='Local Preservation', linewidth=2, markersize=6)
            ax2.axvline(x=best_components, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_components}')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Score')
            ax2.set_title('Distance & Neighborhood Preservation')
            ax2.legend()
            # ax2.grid(True, alpha=0.3)

            # Plot 3: Combined Score
            ax3.plot(df['n_components'], df['combined_score'], 'o-', color='purple', linewidth=3, markersize=8)
            ax3.axvline(x=best_components, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_components}')
            best_idx = df[df['n_components'] == best_components].index[0]
            best_score = df.loc[best_idx, 'combined_score']
            ax3.scatter([best_components], [best_score], color='red', s=150, zorder=5)
            ax3.annotate(f'({best_components}, {best_score:.3f})',
                        xy=(best_components, best_score),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            ax3.set_xlabel('Number of Components')
            ax3.set_ylabel('Combined Score')
            ax3.set_title('Combined Optimization Score')
            ax3.legend()
            # ax3.grid(True, alpha=0.3)

            # Plot 4: All metrics comparison
            metrics = ['trustworthiness', 'continuity', 'procrustes', 'local_preservation']
            for metric in metrics:
                ax4.plot(df['n_components'], df[metric], 'o-', label=metric.replace('_', ' ').title(), linewidth=2, markersize=4)
            ax4.axvline(x=best_components, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {best_components}')
            ax4.set_xlabel('Number of Components')
            ax4.set_ylabel('Score')
            ax4.set_title('All Metrics Comparison')
            ax4.legend()
            # ax4.grid(True, alpha=0.3)

            plt.suptitle(f'{method.upper()}-{device.upper()} Optimization - Fold {fold}\nManifold Learning Metrics Analysis - {level} K-mer {kmer_size}',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save plot using output manager
            plot_filename = f"manifold_optimization_fold{fold}_{best_components}comp_{self.timestamp}.png"
            file_path = self.output_manager.save_file_with_tracking(
                content=plt.gcf(),
                filename=plot_filename,
                level=level,
                kmer=kmer_size,
                method=method,
                device=device,
                file_type='manifold_plot',
                save_func=lambda fig, path: fig.savefig(path, dpi=300, bbox_inches='tight')
            )

            plt.show()
            plt.close()

            return str(file_path) if file_path else None

        except Exception as e:
            print(f"⚠️  Failed to create manifold optimization plot: {str(e)}")
            return None

    def save_features_to_csv(self, X_transformed, y, method, device, fold, optimal_n, level=None, kmer_size=None):
        """Save extracted features to CSV using output manager"""
        try:
            if level is None or kmer_size is None:
                print("⚠️  Level and kmer_size required for organized output")
                return None
                
            filename = f"features_fold{fold}_{optimal_n}comp_{self.timestamp}.csv"

            # Create DataFrame
            feature_cols = [f'feature_{i+1}' for i in range(X_transformed.shape[1])]
            df = pd.DataFrame(X_transformed, columns=feature_cols)
            df['label'] = y
            df['method'] = method
            df['device'] = device
            df['fold'] = fold
            df['n_components'] = optimal_n
            df['level'] = level
            df['kmer_size'] = kmer_size

            # Add metadata
            metadata = [
                f"# Feature Extraction Results",
                f"# Method: {method.upper()}",
                f"# Device: {device.upper()}",
                f"# Fold: {fold}",
                f"# Level: {level}",
                f"# K-mer Size: {kmer_size}",
                f"# Components: {optimal_n}",
                f"# Original shape: {X_transformed.shape}",
                f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"#"
            ]

            # Prepare content with metadata
            content_lines = metadata + ['']  # Add empty line after metadata
            content_lines.append(df.to_csv(index=False))
            content = '\n'.join(content_lines)

            # Save using output manager
            file_path = self.output_manager.save_file_with_tracking(
                content=content,
                filename=filename,
                level=level,
                kmer=kmer_size,
                method=method,
                device=device,
                file_type='features',
                save_func=lambda c, p: p.write_text(c, encoding='utf-8')
            )

            return str(file_path) if file_path else None

        except Exception as e:
            print(f"⚠️  Failed to save features: {str(e)}")
            return None

    def plot_cev_individual(self, cev_values, method, device, fold, optimal_n, level=None, kmer_size=None):
        """Enhanced CEV plotting with output manager"""
        try:
            if level is None or kmer_size is None:
                print("⚠️  Level and kmer_size required for organized output")
                return None
            
            # Ensure cev_values is numpy array for proper operations
            cev_values = np.array(cev_values) if not isinstance(cev_values, np.ndarray) else cev_values
                
            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(cev_values)+1), cev_values,
                    marker='o', markersize=2, linewidth=2, label='CEV')

            plt.axhline(y=self.cev_threshold, color='r', linestyle='--', linewidth=2,
                       label=f"{int(self.cev_threshold*100)}% Threshold")
            plt.axvline(x=optimal_n, color='g', linestyle='--', linewidth=2,
                       label=f"Optimal: {optimal_n} components")

            # Highlight optimal point
            if optimal_n <= len(cev_values):
                plt.scatter([optimal_n], [cev_values[optimal_n-1]],
                          color='red', s=100, zorder=5)
                plt.annotate(f'({optimal_n}, {cev_values[optimal_n-1]*100:.1f}%)',
                           xy=(optimal_n, cev_values[optimal_n-1]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

            plt.title(f'Cumulative Explained Variance\n{method.upper()}-{device.upper()} - Fold {fold} - {level} K-mer {kmer_size}',
                     fontsize=14, fontweight='bold')
            plt.xlabel("Number of Components", fontsize=12)
            plt.ylabel("Cumulative Explained Variance", fontsize=12)
            plt.legend(fontsize=10)
            # plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot using output manager
            plot_filename = f"cev_plot_fold{fold}_{optimal_n}comp_{self.timestamp}.png"
            file_path = self.output_manager.save_file_with_tracking(
                content=plt.gcf(),
                filename=plot_filename,
                level=level,
                kmer=kmer_size,
                method=method,
                device=device,
                file_type='cev_plot',
                save_func=lambda fig, path: fig.savefig(path, dpi=300, bbox_inches='tight')
            )

            plt.show()
            plt.close()

            return str(file_path) if file_path else None

        except Exception as e:
            print(f"⚠️  Failed to create CEV plot: {str(e)}")
            return None

    def save_cev_to_csv(self, cev_values, explained_variance_ratio, method, device, fold, optimal_n, level=None, kmer_size=None):
        """Enhanced CEV CSV saving with output manager"""
        try:
            if level is None or kmer_size is None:
                print("⚠️  Level and kmer_size required for organized output")
                return None
                
            filename = f"cev_analysis_fold{fold}_{optimal_n}components_{self.timestamp}.csv"

            # Ensure cev_values is numpy array for proper operations
            cev_values = np.array(cev_values) if not isinstance(cev_values, np.ndarray) else cev_values
            explained_variance_ratio = np.array(explained_variance_ratio) if not isinstance(explained_variance_ratio, np.ndarray) else explained_variance_ratio
            
            n_components = len(cev_values)
            data = {
                'component_number': range(1, n_components + 1),
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_explained_variance': cev_values,
                'variance_percentage': cev_values * 100
            }

            threshold_reached = np.where(cev_values >= self.cev_threshold)[0]
            threshold_component = threshold_reached[0] + 1 if len(threshold_reached) > 0 else None

            df = pd.DataFrame(data)

            metadata = [
                f"# CEV Analysis Results",
                f"# Method: {method.upper()}",
                f"# Device: {device.upper()}",
                f"# Fold: {fold}",
                f"# Level: {level}",
                f"# K-mer Size: {kmer_size}",
                f"# CEV Threshold: {self.cev_threshold*100:.1f}%",
                f"# Optimal Components: {optimal_n}",
                f"# Threshold Reached at Component: {threshold_component if threshold_component else 'Not reached'}",
                f"# Final CEV: {cev_values[-1]*100:.2f}%",
                f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"#"
            ]

            # Prepare content with metadata
            content_lines = metadata + ['']  # Add empty line after metadata
            content_lines.append(df.to_csv(index=False))
            content = '\n'.join(content_lines)

            # Save using output manager
            file_path = self.output_manager.save_file_with_tracking(
                content=content,
                filename=filename,
                level=level,
                kmer=kmer_size,
                method=method,
                device=device,
                file_type='cev_analysis',
                save_func=lambda c, p: p.write_text(c, encoding='utf-8')
            )

            return str(file_path) if file_path else None

        except Exception as e:
            print(f"⚠️  Failed to save CEV data: {str(e)}")
            return None

    # =================================================================
    # TRADITIONAL METHODS (PCA, SVD)
    # =================================================================

    def batch_svd_cpu(self, X_sparse, y, n_components, random_state=42):
        """Enhanced TruncatedSVD dengan stratified sampling"""
        self.check_memory_usage("SVD CPU Start")

        batch_size = self.adaptive_batch_size(n_components, X_sparse.shape)
        n_samples = min(batch_size * 2, X_sparse.shape[0], 2000)

        # Use stratified sampling
        X_sample, y_sample, _ = self.create_stratified_sample(X_sparse, y, n_samples, random_state)
        X_sample_dense = safe_to_dense(X_sample)

        U, s, Vt = randomized_svd(X_sample_dense, n_components=n_components, random_state=random_state)

        explained_variance = s**2 / (n_samples - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance

        del X_sample_dense, U, s, explained_variance
        gc.collect()

        self.check_memory_usage("SVD CPU End")
        return Vt, explained_variance_ratio

    def transform_batch_svd_cpu(self, X_sparse, Vt):
        """Enhanced SVD transform with memory management"""
        n_batches = int(np.ceil(X_sparse.shape[0] / self.batch_size))
        X_transformed = []

        for i in tqdm(range(n_batches), desc="SVD Transform (CPU)"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, X_sparse.shape[0])
            batch = safe_slice_to_dense(X_sparse, start_idx, end_idx)
            batch_transformed = batch @ Vt.T
            X_transformed.append(batch_transformed)

            del batch, batch_transformed

            if i % 3 == 0:
                gc.collect()
                self.check_memory_usage(f"SVD Transform batch {i}")

        result = np.vstack(X_transformed)
        del X_transformed
        gc.collect()
        return result

    def svd_gpu(self, X_sparse, y, n_components):
        """Enhanced TruncatedSVD GPU with better error handling"""
        if not CUML_AVAILABLE:
            raise RuntimeError("GPU tidak tersedia")

        self.check_memory_usage("SVD GPU Start")

        try:
            print(f"🔧 Converting sparse matrix to float32 for GPU compatibility...")

            if X_sparse.shape[0] > 10000:
                sample_size = min(10000, X_sparse.shape[0])
                X_sample, y_sample, _ = self.create_stratified_sample(X_sparse, y, sample_size)

                coo = X_sample.tocoo()
                data_gpu = cp.asarray(coo.data, dtype=cp.float32)
                row_gpu = cp.asarray(coo.row, dtype=cp.int32)
                col_gpu = cp.asarray(coo.col, dtype=cp.int32)

                X_gpu = cp.sparse.csr_matrix((data_gpu, (row_gpu, col_gpu)),
                                           shape=coo.shape, dtype=cp.float32)
            else:
                coo = X_sparse.tocoo()
                data_gpu = cp.asarray(coo.data, dtype=cp.float32)
                row_gpu = cp.asarray(coo.row, dtype=cp.int32)
                col_gpu = cp.asarray(coo.col, dtype=cp.int32)

                X_gpu = cp.sparse.csr_matrix((data_gpu, (row_gpu, col_gpu)),
                                           shape=coo.shape, dtype=cp.float32)

            print(f"🔧 GPU matrix dtype: {X_gpu.dtype}")

            model = CuTruncatedSVD(n_components=n_components, random_state=42)
            model.fit(X_gpu)

            explained_variance_ratio = model.explained_variance_ratio_.get()

            # Cleanup GPU memory
            del X_gpu, data_gpu, row_gpu, col_gpu, coo
            self.check_gpu_memory()
            gc.collect()

            self.check_memory_usage("SVD GPU End")
            return model, explained_variance_ratio

        except Exception as e:
            print(f"🔧 GPU SVD Error: {str(e)}")
            self.check_gpu_memory()
            gc.collect()
            raise e

    def ipca_cpu(self, X_sparse, n_components):
        """Enhanced IncrementalPCA CPU"""
        self.check_memory_usage("IPCA CPU Start")

        adaptive_batch = self.adaptive_batch_size(n_components, X_sparse.shape)
        print(f"🔧 IPCA using adaptive batch size: {adaptive_batch}")

        model = IncrementalPCA(n_components=n_components, whiten=True)
        n_batches = int(np.ceil(X_sparse.shape[0] / adaptive_batch))

        for i in tqdm(range(n_batches), desc="IPCA Fit (CPU)"):
            start_idx = i * adaptive_batch
            end_idx = min((i + 1) * adaptive_batch, X_sparse.shape[0])
            batch = safe_slice_to_dense(X_sparse, start_idx, end_idx)

            if i == 0 and batch.shape[0] < n_components:
                end_idx = min(start_idx + n_components + 50, X_sparse.shape[0])
                batch = safe_slice_to_dense(X_sparse, start_idx, end_idx)
                print(f"🔧 Adjusted first batch size to {batch.shape[0]} samples")

            model.partial_fit(batch)
            del batch

            if i % 3 == 0:
                gc.collect()
                self.check_memory_usage(f"IPCA Fit batch {i}")

        self.check_memory_usage("IPCA CPU End")
        return model, model.explained_variance_ratio_

    def ipca_gpu(self, X_sparse, n_components):
        """Enhanced IncrementalPCA GPU"""
        if not CUML_AVAILABLE:
            raise RuntimeError("GPU tidak tersedia")

        self.check_memory_usage("IPCA GPU Start")

        try:
            adaptive_batch = self.adaptive_batch_size(n_components, X_sparse.shape)

            if X_sparse.shape[0] > 5000:
                model = CuIncrementalPCA(n_components=n_components)
                n_batches = int(np.ceil(X_sparse.shape[0] / adaptive_batch))

                print(f"🔧 IPCA GPU using adaptive batch size: {adaptive_batch}")

                for i in tqdm(range(n_batches), desc="IPCA GPU Fit"):
                    start_idx = i * adaptive_batch
                    end_idx = min((i + 1) * adaptive_batch, X_sparse.shape[0])
                    batch = X_sparse[start_idx:end_idx].toarray()

                    batch_gpu = cp.asarray(batch, dtype=cp.float32)

                    if i == 0 and batch.shape[0] < n_components:
                        end_idx = min(start_idx + n_components + 50, X_sparse.shape[0])
                        batch = X_sparse[start_idx:end_idx].toarray()
                        batch_gpu = cp.asarray(batch, dtype=cp.float32)
                        print(f"🔧 Adjusted first GPU batch size to {batch.shape[0]} samples")

                    if i == 0:
                        model.fit(batch_gpu)
                    else:
                        try:
                            model.partial_fit(batch_gpu)
                        except:
                            print(f"⚠️  CuML partial_fit not supported, using fallback method")
                            break

                    del batch, batch_gpu
                    if i % 3 == 0:
                        self.check_gpu_memory()
                        gc.collect()
            else:
                X_dense = X_sparse.toarray()
                X_gpu = cp.asarray(X_dense, dtype=cp.float32)
                model = CuIncrementalPCA(n_components=n_components)
                model.fit(X_gpu)
                del X_dense

            explained_variance_ratio = model.explained_variance_ratio_.get()

            self.check_gpu_memory()
            gc.collect()

            self.check_memory_usage("IPCA GPU End")
            return model, explained_variance_ratio

        except Exception as e:
            print(f"🔧 GPU IPCA Error: {str(e)}")
            self.check_gpu_memory()
            gc.collect()
            raise e

    # =================================================================
    # UMAP IMPLEMENTATION
    # =================================================================

    def umap_cpu(self, X_sparse, n_components):
        """UMAP implementation for CPU"""
        if not UMAP_AVAILABLE:
            raise RuntimeError("UMAP tidak tersedia")

        self.check_memory_usage("UMAP CPU Start")

        try:
            # Convert to dense for UMAP
            if X_sparse.shape[0] > 5000:
                # Sample for large datasets
                sample_indices = np.random.choice(X_sparse.shape[0], 5000, replace=False)
                X_sample = X_sparse[sample_indices].toarray()
            else:
                X_sample = X_sparse.toarray()

            # Fit UMAP
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=self.umap_n_neighbors,
                random_state=42,
                verbose=True
            )

            print(f"🔧 Fitting UMAP with {X_sample.shape[0]} samples...")
            model.fit(X_sample)

            # UMAP doesn't have explained_variance_ratio, so we'll estimate it
            # by comparing variance before and after transformation
            X_transformed_sample = model.transform(X_sample)

            original_var = np.var(X_sample, axis=0)
            transformed_var = np.var(X_transformed_sample, axis=0)

            # Estimate explained variance ratio
            explained_variance_ratio = transformed_var / np.sum(original_var)
            explained_variance_ratio = explained_variance_ratio / np.sum(explained_variance_ratio)

            del X_sample, X_transformed_sample
            gc.collect()

            self.check_memory_usage("UMAP CPU End")
            return model, explained_variance_ratio

        except Exception as e:
            print(f"🔧 UMAP CPU Error: {str(e)}")
            gc.collect()
            raise e

    def umap_gpu(self, X_sparse, n_components):
        """UMAP implementation for GPU using cuML"""
        if not CUML_AVAILABLE:
            raise RuntimeError("cuML UMAP tidak tersedia")

        self.check_memory_usage("UMAP GPU Start")

        try:
            import cuml.manifold.umap as cu_umap

            # Convert to GPU format
            if X_sparse.shape[0] > 5000:
                sample_indices = np.random.choice(X_sparse.shape[0], 5000, replace=False)
                X_sample = X_sparse[sample_indices].toarray()
            else:
                X_sample = X_sparse.toarray()

            X_gpu = cp.asarray(X_sample, dtype=cp.float32)

            # Fit cuML UMAP
            model = cu_umap.UMAP(
                n_components=n_components,
                n_neighbors=self.umap_n_neighbors,
                random_state=42
            )

            print(f"🔧 Fitting cuML UMAP with {X_sample.shape[0]} samples...")
            model.fit(X_gpu)

            # Transform to estimate explained variance
            X_transformed_gpu = model.transform(X_gpu)
            X_transformed = X_transformed_gpu.get()

            original_var = np.var(X_sample, axis=0)
            transformed_var = np.var(X_transformed, axis=0)

            explained_variance_ratio = transformed_var / np.sum(original_var)
            explained_variance_ratio = explained_variance_ratio / np.sum(explained_variance_ratio)

            del X_sample, X_gpu, X_transformed_gpu, X_transformed
            self.check_gpu_memory()
            gc.collect()

            self.check_memory_usage("UMAP GPU End")
            return model, explained_variance_ratio

        except Exception as e:
            print(f"🔧 UMAP GPU Error: {str(e)}")
            self.check_gpu_memory()
            gc.collect()
            raise e

    # =================================================================
    # AUTOENCODER IMPLEMENTATION
    # =================================================================

    def autoencoder_cpu(self, X_sparse, n_components):
        """Autoencoder implementation for CPU"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch tidak tersedia")

        self.check_memory_usage("Autoencoder CPU Start")

        try:
            device = torch.device('cpu')

            # Prepare data
            X_dense = X_sparse.toarray().astype(np.float32)

            # Normalize data
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X_dense)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_normalized).to(device)

            # Create model
            input_dim = X_tensor.shape[1]
            model = SimpleAutoencoder(input_dim, n_components).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=min(512, len(X_tensor)), shuffle=True)

            print(f"🔧 Training Autoencoder for {self.autoencoder_epochs} epochs...")

            model.train()
            for epoch in tqdm(range(self.autoencoder_epochs), desc="Training Autoencoder"):
                total_loss = 0
                for batch in dataloader:
                    batch_data = batch[0]

                    optimizer.zero_grad()
                    encoded, decoded = model(batch_data)
                    loss = criterion(decoded, batch_data)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            # Get encoded representation
            model.eval()
            with torch.no_grad():
                encoded_data = model.encode(X_tensor).cpu().numpy()

            # Estimate explained variance ratio
            original_var = np.var(X_normalized, axis=0)
            encoded_var = np.var(encoded_data, axis=0)

            explained_variance_ratio = encoded_var / np.sum(original_var)
            explained_variance_ratio = explained_variance_ratio / np.sum(explained_variance_ratio)

            # Store the model and scaler for later use
            model_dict = {
                'model': model,
                'scaler': scaler,
                'device': device
            }

            del X_dense, X_normalized, X_tensor, encoded_data
            gc.collect()

            self.check_memory_usage("Autoencoder CPU End")
            return model_dict, explained_variance_ratio

        except Exception as e:
            print(f"🔧 Autoencoder CPU Error: {str(e)}")
            gc.collect()
            raise e

    def autoencoder_gpu(self, X_sparse, n_components):
        """Autoencoder implementation for GPU"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("PyTorch GPU tidak tersedia")

        self.check_memory_usage("Autoencoder GPU Start")

        try:
            device = torch.device('cuda')

            # Prepare data
            X_dense = X_sparse.toarray().astype(np.float32)

            # Normalize data
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X_dense)

            # Convert to tensor
            X_tensor = torch.FloatTensor(X_normalized).to(device)

            # Create model
            input_dim = X_tensor.shape[1]
            model = SimpleAutoencoder(input_dim, n_components).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training
            dataset = TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=min(512, len(X_tensor)), shuffle=True)

            print(f"🔧 Training Autoencoder on GPU for {self.autoencoder_epochs} epochs...")

            model.train()
            for epoch in tqdm(range(self.autoencoder_epochs), desc="Training Autoencoder GPU"):
                total_loss = 0
                for batch in dataloader:
                    batch_data = batch[0]

                    optimizer.zero_grad()
                    encoded, decoded = model(batch_data)
                    loss = criterion(decoded, batch_data)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % 10 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

                # Clear cache periodically
                if epoch % 20 == 0:
                    torch.cuda.empty_cache()

            # Get encoded representation
            model.eval()
            with torch.no_grad():
                encoded_data = model.encode(X_tensor).cpu().numpy()

            # Estimate explained variance ratio
            original_var = np.var(X_normalized, axis=0)
            encoded_var = np.var(encoded_data, axis=0)

            explained_variance_ratio = encoded_var / np.sum(original_var)
            explained_variance_ratio = explained_variance_ratio / np.sum(explained_variance_ratio)

            # Store the model and scaler for later use
            model_dict = {
                'model': model,
                'scaler': scaler,
                'device': device
            }

            del X_dense, X_normalized, X_tensor, encoded_data
            torch.cuda.empty_cache()
            gc.collect()

            self.check_memory_usage("Autoencoder GPU End")
            return model_dict, explained_variance_ratio

        except Exception as e:
            print(f"🔧 Autoencoder GPU Error: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            raise e

    # =================================================================
    # OPTIMAL COMPONENT FINDING
    # =================================================================

    def find_optimal_components(self, X_sparse, y, method, device, fold, level=None, kmer_size=None):
        """Enhanced optimal component finding with proper data size validation"""

        # Validate data size first
        n_samples, n_features = X_sparse.shape
        print(f"📊 Data validation: {n_samples} samples, {n_features} features")
        
        # Calculate maximum safe components
        max_possible_components = min(n_samples - 10, n_features - 10, self.max_limit)
        
        # Minimum components for meaningful analysis
        min_components = max(2, min(5, n_samples // 10))
        
        if max_possible_components < min_components:
            print(f"⚠️  Data too small for analysis: max_possible={max_possible_components}, min_required={min_components}")
            # Return minimal components for very small datasets
            safe_components = max(2, min(n_samples // 2, n_features // 2))
            print(f"🔧 Using minimal safe components: {safe_components}")
            
            # Create dummy results for very small data
            dummy_cev = [0.8]  # Single high CEV value
            dummy_explained_ratio = [0.8]
            
            cev_csv_path = self.save_cev_to_csv(
                dummy_cev, dummy_explained_ratio, method, device, fold, safe_components, level, kmer_size
            )
            cev_plot_path = self.plot_cev_individual(dummy_cev, method, device, fold, safe_components, level, kmer_size)
            
            return safe_components, 0.8, cev_csv_path, cev_plot_path

        # Special handling for UMAP and Autoencoder
        if method in ['umap', 'autoencoder']:
            return self.find_optimal_components_manifold(X_sparse, y, method, device, fold, level, kmer_size)

        # Traditional methods (PCA, SVD, IPCA) - Enhanced validation
        start_components = min(self.start_components, max_possible_components)
        print(f"🔧 Safe component range: {min_components} - {max_possible_components}")
        print(f"🔧 Starting with: {start_components} components")

        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"🔍 Iteration {iteration}: Testing {method.upper()}-{device.upper()} with {start_components} components...")

            # Method-specific validation
            if method == 'ipca':
                # IPCA needs sufficient batch size
                min_batch_for_ipca = start_components + 20
                if n_samples < min_batch_for_ipca:
                    # Reduce components to fit data
                    start_components = max(min_components, n_samples - 20)
                    print(f"🔧 IPCA: Reduced components to {start_components} for data size {n_samples}")
                    
                    if start_components < min_components:
                        print(f"❌ Data too small for IPCA: need at least {min_batch_for_ipca} samples")
                        # Fall back to minimal solution
                        safe_components = 2
                        dummy_cev = [0.7]
                        dummy_explained_ratio = [0.7]
                        
                        cev_csv_path = self.save_cev_to_csv(
                            dummy_cev, dummy_explained_ratio, method, device, fold, safe_components, level, kmer_size
                        )
                        cev_plot_path = self.plot_cev_individual(dummy_cev, method, device, fold, safe_components, level, kmer_size)
                        
                        return safe_components, 0.7, cev_csv_path, cev_plot_path

            # Validate components one more time
            if start_components > max_possible_components:
                start_components = max_possible_components
                print(f"� Clamped components to maximum: {start_components}")

            if start_components < min_components:
                start_components = min_components
                print(f"🔧 Increased components to minimum: {start_components}")

            self.check_memory_usage(f"{method}-{device} search")

            try:
                if method == 'svd' and device == 'cpu':
                    _, explained_variance_ratio = self.batch_svd_cpu(X_sparse, y, start_components)
                elif method == 'svd' and device == 'gpu':
                    _, explained_variance_ratio = self.svd_gpu(X_sparse, y, start_components)
                elif method == 'ipca' and device == 'cpu':
                    _, explained_variance_ratio = self.ipca_cpu(X_sparse, start_components)
                elif method == 'ipca' and device == 'gpu':
                    _, explained_variance_ratio = self.ipca_gpu(X_sparse, start_components)
                else:
                    raise ValueError(f"Invalid combination: {method}-{device}")

                cev = np.cumsum(explained_variance_ratio)

                if np.any(cev >= self.cev_threshold):
                    optimal_n = max(min_components, np.argmax(cev >= self.cev_threshold) + 1)
                    final_cev = cev[optimal_n-1]

                    cev_csv_path = self.save_cev_to_csv(
                        cev, explained_variance_ratio, method, device, fold, optimal_n, level, kmer_size
                    )
                    cev_plot_path = self.plot_cev_individual(cev, method, device, fold, optimal_n, level, kmer_size)

                    del explained_variance_ratio, cev
                    gc.collect()

                    return optimal_n, final_cev, cev_csv_path, cev_plot_path
                else:
                    print(f"⚠️  CEV threshold not reached with {start_components} components")
                    print(f"   Max CEV achieved: {cev[-1]:.4f}")
                    
                    # Check if we can increase components
                    if start_components < max_possible_components:
                        new_components = min(start_components + self.step_components, max_possible_components)
                        print(f"🔧 Increasing components: {start_components} → {new_components}")
                        start_components = new_components
                        
                        del explained_variance_ratio, cev
                        gc.collect()
                        continue
                    else:
                        # Use maximum available components
                        print(f"🔧 Using maximum components: {start_components}")
                        optimal_n = start_components
                        final_cev = cev[-1]

                        cev_csv_path = self.save_cev_to_csv(
                            cev, explained_variance_ratio, method, device, fold, optimal_n, level, kmer_size
                        )
                        cev_plot_path = self.plot_cev_individual(cev, method, device, fold, optimal_n, level, kmer_size)

                        del explained_variance_ratio, cev
                        gc.collect()

                        return optimal_n, final_cev, cev_csv_path, cev_plot_path

            except Exception as e:
                print(f"❌ Error with {start_components} components: {str(e)}")
                
                # Try with smaller components
                if start_components > min_components:
                    start_components = max(min_components, start_components // 2)
                    print(f"🔧 Retrying with reduced components: {start_components}")
                    continue
                else:
                    print(f"❌ Failed even with minimum components, using fallback")
                    # Ultimate fallback
                    safe_components = 2
                    dummy_cev = np.array([0.6])  # Convert to numpy array
                    dummy_explained_ratio = np.array([0.6])  # Convert to numpy array
                    
                    cev_csv_path = self.save_cev_to_csv(
                        dummy_cev, dummy_explained_ratio, method, device, fold, safe_components, level, kmer_size
                    )
                    cev_plot_path = self.plot_cev_individual(dummy_cev, method, device, fold, safe_components, level, kmer_size)
                    
                    return safe_components, 0.6, cev_csv_path, cev_plot_path

        # If we exit the loop without success, return fallback
        print(f"⚠️  Maximum iterations reached, using fallback solution")
        safe_components = min_components
        dummy_cev = np.array([0.5])  # Convert to numpy array
        dummy_explained_ratio = np.array([0.5])  # Convert to numpy array
        
        cev_csv_path = self.save_cev_to_csv(
            dummy_cev, dummy_explained_ratio, method, device, fold, safe_components, level, kmer_size
        )
        cev_plot_path = self.plot_cev_individual(dummy_cev, method, device, fold, safe_components, level, kmer_size)
        
        return safe_components, 0.5, cev_csv_path, cev_plot_path

    def find_optimal_components_manifold(self, X_sparse, y, method, device, fold, level=None, kmer_size=None):
        """
        Enhanced optimal component finding for UMAP/Autoencoder using manifold learning metrics
        """
        # Define component ranges to test
        if method == 'umap':
            components_to_try = [2, 3, 5, 8, 10, 15, 20, 30, 50]
        elif method == 'autoencoder':
            components_to_try = [5, 10, 20, 30, 50, 100, 150, 200]
        else:
            components_to_try = [2, 5, 10, 20, 50]

        # Filter components based on data constraints
        max_possible = min(X_sparse.shape[0] // 2, X_sparse.shape[1] // 2)
        components_to_try = [c for c in components_to_try if c < max_possible]

        if not components_to_try:
            components_to_try = [2, min(5, max_possible-1)]

        print(f"🔍 Testing {method.upper()}-{device.upper()} with components: {components_to_try}")

        # Prepare data
        X_dense = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else X_sparse

        # Sample for large datasets to avoid memory issues
        sample_size = min(2000, len(X_dense))
        if len(X_dense) > sample_size:
            sample_indices = np.random.choice(len(X_dense), sample_size, replace=False)
            X_sample = X_dense[sample_indices]
            y_sample = y[sample_indices] if y is not None else None
        else:
            X_sample = X_dense
            y_sample = y

        results = []
        best_score = -1
        best_components = components_to_try[0]
        best_features = None

        for n_comp in components_to_try:
            try:
                print(f"   🧪 Testing {n_comp} components...")

                # Transform data using the specific method
                X_embedded = self.transform_data_direct(
                    X_sparse[sample_indices] if len(X_dense) > sample_size else X_sparse,
                    method, device, n_comp
                )

                # Calculate manifold learning metrics
                trust, cont = self.calculate_trustworthiness_continuity(X_sample, X_embedded)
                procrustes = self.calculate_procrustes_score(X_sample, X_embedded)
                local_preserve = self.calculate_local_neighborhood_preservation(X_sample, X_embedded)
                intrinsic_dim = self.estimate_intrinsic_dimension(X_embedded)

                # Calculate silhouette if labels available
                silhouette = self.calculate_silhouette_score(X_embedded, y_sample) if y_sample is not None else 0

                # Calculate combined score with weights
                weights = {
                    'trustworthiness': 0.35,
                    'continuity': 0.35,
                    'procrustes': 0.15,
                    'local_preservation': 0.15
                }

                combined_score = (
                    weights['trustworthiness'] * trust +
                    weights['continuity'] * cont +
                    weights['procrustes'] * max(0, procrustes) +  # Handle NaN
                    weights['local_preservation'] * local_preserve
                )

                result = {
                    'n_components': n_comp,
                    'trustworthiness': trust,
                    'continuity': cont,
                    'procrustes': procrustes,
                    'local_preservation': local_preserve,
                    'silhouette': silhouette,
                    'intrinsic_dimension': intrinsic_dim,
                    'combined_score': combined_score
                }

                results.append(result)

                print(f"      Trust: {trust:.4f}, Cont: {cont:.4f}")
                print(f"      Proc: {procrustes:.4f}, Local: {local_preserve:.4f}")
                print(f"      Combined: {combined_score:.4f}")

                # Update best result
                if combined_score > best_score:
                    best_score = combined_score
                    best_components = n_comp
                    best_features = X_embedded.copy()

                del X_embedded
                gc.collect()
                self.check_memory_usage(f"Manifold test {n_comp}")

            except Exception as e:
                print(f"⚠️  Failed with {n_comp} components: {str(e)}")
                continue

        if not results:
            print("❌ All component tests failed!")
            return 2, 0.0, None, None

        # Find the actual best result
        best_result = max(results, key=lambda x: x['combined_score'])
        best_components = best_result['n_components']
        best_score = best_result['combined_score']

        print(f"🎯 Optimal {method.upper()} components: {best_components}")
        print(f"   Best combined score: {best_score:.4f}")
        print(f"   Trustworthiness: {best_result['trustworthiness']:.4f}")
        print(f"   Continuity: {best_result['continuity']:.4f}")

        # Save detailed results
        results_path = self.save_manifold_optimization_results(results, method, device, fold, level, kmer_size)

        # Create optimization plot
        plot_path = self.plot_manifold_optimization(results, method, device, fold, best_components, level, kmer_size)

        # Store additional metrics for later use
        self.manifold_metrics = best_result

        return best_components, best_score, results_path, plot_path

    # =================================================================
    # TRANSFORM DATA METHODS
    # =================================================================

    def transform_data_direct(self, X_sparse, method, device, n_components):
        """Direct transform for methods without CEV optimization"""
        if method == 'umap' and device == 'cpu':
            model, _ = self.umap_cpu(X_sparse, n_components)
            return model.transform(X_sparse.toarray())
        elif method == 'umap' and device == 'gpu':
            model, _ = self.umap_gpu(X_sparse, n_components)
            X_gpu = cp.asarray(X_sparse.toarray(), dtype=cp.float32)
            result = model.transform(X_gpu).get()
            del X_gpu
            self.check_gpu_memory()
            return result
        elif method == 'autoencoder' and device == 'cpu':
            model_dict, _ = self.autoencoder_cpu(X_sparse, n_components)
            X_dense = X_sparse.toarray().astype(np.float32)
            X_normalized = model_dict['scaler'].transform(X_dense)
            X_tensor = torch.FloatTensor(X_normalized).to(model_dict['device'])
            with torch.no_grad():
                result = model_dict['model'].encode(X_tensor).cpu().numpy()
            del X_dense, X_normalized, X_tensor
            return result
        elif method == 'autoencoder' and device == 'gpu':
            model_dict, _ = self.autoencoder_gpu(X_sparse, n_components)
            X_dense = X_sparse.toarray().astype(np.float32)
            X_normalized = model_dict['scaler'].transform(X_dense)
            X_tensor = torch.FloatTensor(X_normalized).to(model_dict['device'])
            with torch.no_grad():
                result = model_dict['model'].encode(X_tensor).cpu().numpy()
            del X_dense, X_normalized, X_tensor
            torch.cuda.empty_cache()
            return result
        else:
            raise ValueError(f"Unknown method: {method}-{device}")

    def transform_data(self, X_sparse, y, method, device, n_components):
        """Enhanced transform data with all methods"""
        self.check_memory_usage(f"Transform {method}-{device} start")

        try:
            if method == 'svd' and device == 'cpu':
                Vt, _ = self.batch_svd_cpu(X_sparse, y, n_components)
                X_transformed = self.transform_batch_svd_cpu(X_sparse, Vt)
                del Vt

            elif method == 'svd' and device == 'gpu':
                model, _ = self.svd_gpu(X_sparse, y, n_components)
                coo = X_sparse.tocoo()
                data_gpu = cp.asarray(coo.data, dtype=cp.float32)
                row_gpu = cp.asarray(coo.row, dtype=cp.int32)
                col_gpu = cp.asarray(coo.col, dtype=cp.int32)

                X_gpu = cp.sparse.csr_matrix((data_gpu, (row_gpu, col_gpu)),
                                           shape=coo.shape, dtype=cp.float32)

                X_transformed = model.transform(X_gpu).get()
                del X_gpu, data_gpu, row_gpu, col_gpu, coo
                self.check_gpu_memory()

            elif method == 'ipca' and device == 'cpu':
                model, _ = self.ipca_cpu(X_sparse, n_components)
                adaptive_batch = self.adaptive_batch_size(n_components, X_sparse.shape)
                X_transformed = []
                n_batches = int(np.ceil(X_sparse.shape[0] / adaptive_batch))

                for i in tqdm(range(n_batches), desc="IPCA Transform (CPU)"):
                    start_idx = i * adaptive_batch
                    end_idx = min((i + 1) * adaptive_batch, X_sparse.shape[0])
                    batch = X_sparse[start_idx:end_idx].toarray()
                    transformed = model.transform(batch)
                    X_transformed.append(transformed)
                    del batch, transformed
                    if i % 3 == 0:
                        gc.collect()

                X_transformed = np.vstack(X_transformed)

            elif method == 'ipca' and device == 'gpu':
                model, _ = self.ipca_gpu(X_sparse, n_components)

                if X_sparse.shape[0] > 5000:
                    adaptive_batch = self.adaptive_batch_size(n_components, X_sparse.shape)
                    X_transformed = []
                    n_batches = int(np.ceil(X_sparse.shape[0] / adaptive_batch))

                    for i in tqdm(range(n_batches), desc="IPCA Transform (GPU)"):
                        start_idx = i * adaptive_batch
                        end_idx = min((i + 1) * adaptive_batch, X_sparse.shape[0])
                        batch = X_sparse[start_idx:end_idx].toarray()

                        batch_gpu = cp.asarray(batch, dtype=cp.float32)
                        transformed = model.transform(batch_gpu).get()
                        X_transformed.append(transformed)
                        del batch, batch_gpu, transformed
                        if i % 3 == 0:
                            self.check_gpu_memory()
                            gc.collect()
                    X_transformed = np.vstack(X_transformed)
                else:
                    X_dense = X_sparse.toarray()
                    X_gpu = cp.asarray(X_dense, dtype=cp.float32)
                    X_transformed = model.transform(X_gpu).get()
                    del X_dense, X_gpu
                    self.check_gpu_memory()

            elif method == 'umap':
                X_transformed = self.transform_data_direct(X_sparse, method, device, n_components)

            elif method == 'autoencoder':
                X_transformed = self.transform_data_direct(X_sparse, method, device, n_components)

            else:
                raise ValueError(f"Unknown method: {method}-{device}")

            gc.collect()
            self.check_memory_usage(f"Transform {method}-{device} end")
            return X_transformed

        except Exception as e:
            gc.collect()
            if device == 'gpu':
                self.check_gpu_memory()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e

    def get_available_combinations(self):
        """Get available method-device combinations with robust GPU checking"""
        combinations = [
            ('ipca', 'cpu'),
            ('svd', 'cpu'),
        ]

        # Safe GPU checks
        gpu_available = False

        if CUML_AVAILABLE:
            try:
                # Test CuML functionality
                import cupy as cp
                cp.cuda.runtime.getDeviceCount()
                gpu_available = True
                combinations.extend([
                    ('ipca', 'gpu'),
                    ('svd', 'gpu'),
                ])
                print("✅ CuML GPU methods available")
            except Exception as e:
                print(f"⚠️ CuML GPU test failed: {str(e)}")

        if UMAP_AVAILABLE:
            combinations.append(('umap', 'cpu'))
            if gpu_available:
                try:
                    import cuml.manifold.umap
                    combinations.append(('umap', 'gpu'))
                    print("✅ UMAP GPU available")
                except Exception as e:
                    print(f"⚠️ UMAP GPU not available: {str(e)}")

        if TORCH_AVAILABLE:
            combinations.append(('autoencoder', 'cpu'))
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0]).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    combinations.append(('autoencoder', 'gpu'))
                    print("✅ PyTorch GPU available")
                except Exception as e:
                    print(f"⚠️ PyTorch GPU test failed: {str(e)}")

        print(f"🔧 Available combinations: {len(combinations)}")
        for method, device in combinations:
            print(f"   - {method.upper()}-{device.upper()}")

        return combinations

class ProductionDimensionalityBenchmark(EnhancedDimensionalityReductionBenchmark):
    """Production-ready benchmark dengan multi-level taxonomic dan k-mer analysis"""

    def __init__(self, cev_threshold=0.95, start_components=100, step_components=50,
                 max_limit=1000, batch_size=500, cv_folds=3, autoencoder_epochs=50,
                 umap_n_neighbors=15, taxonomic_levels=['genus', 'species', 'family', 'order', 'phylum'],
                 kmer_sizes=[6, 8, 10], enable_manifold_metrics=True,
                 output_directory=None, skip_existing=True):

        super().__init__(cev_threshold, start_components, step_components, max_limit,
                        batch_size, cv_folds, autoencoder_epochs, umap_n_neighbors,
                        taxonomic_levels, kmer_sizes, enable_manifold_metrics,
                        output_directory, skip_existing)  # Pass output parameters to parent

        self.taxonomic_levels = taxonomic_levels
        self.kmer_sizes = kmer_sizes
        self.grid_search_results = []
        self.all_results_df = None
        self.production_features = {}  # Store features for each combination

        print(f"🏭 PRODUCTION MODE INITIALIZED:")
        print(f"   - Taxonomic levels: {self.taxonomic_levels}")
        print(f"   - K-mer sizes: {self.kmer_sizes}")
        print(f"   - Total combinations: {len(self.taxonomic_levels) * len(self.kmer_sizes)}")
        print(f"   - Output directory: {self.output_manager.base_output_path}")
        print(f"   - Skip existing: {self.output_manager.skip_existing}")



    # =================================================================
    # GRID SEARCH IMPLEMENTATION
    # =================================================================

    def define_parameter_grid(self):
        """Define comprehensive parameter grid for optimization"""
        param_grid = {
            'cev_threshold': [0.90, 0.95, 0.98],
            'batch_size': [200, 500, 1000],
            'start_components': [50, 100, 200],
            'umap_n_neighbors': [5, 15, 30],
            'autoencoder_epochs': [30, 50, 100]
        }
        return param_grid

    def grid_search_single_combination(self, X_train_sparse, y_train, level, kmer_size):
        """Perform grid search for a single level-kmer combination"""
        param_grid = self.define_parameter_grid()
        grid = ParameterGrid(param_grid)

        print(f"\n🔍 GRID SEARCH: {level.upper()} - K-mer {kmer_size}")
        print(f"   Testing {len(grid)} parameter combinations...")

        best_params = None
        best_score = -np.inf
        grid_results = []

        for i, params in enumerate(grid):
            print(f"\n📊 Testing parameter set {i+1}/{len(grid)}")
            print(f"   Params: {params}")

            try:
                # Create temporary benchmark with these parameters
                temp_benchmark = ProductionDimensionalityBenchmark(
                    cev_threshold=params['cev_threshold'],
                    start_components=params['start_components'],
                    batch_size=params['batch_size'],
                    cv_folds=self.cv_folds,
                    autoencoder_epochs=params['autoencoder_epochs'],
                    umap_n_neighbors=params['umap_n_neighbors'],
                    taxonomic_levels=[level],
                    kmer_sizes=[kmer_size]
                )

                # Run quick benchmark (limited methods for speed)
                quick_combinations = [('ipca', 'cpu'), ('svd', 'cpu')]
                if UMAP_AVAILABLE:
                    quick_combinations.append(('umap', 'cpu'))

                param_results = []

                for method, device in quick_combinations:
                    try:
                        result = temp_benchmark.run_single_benchmark_fold(
                            X_train_sparse, y_train, method, device, level, kmer_size, fold=0
                        )
                        if result.success:
                            param_results.append(result)
                    except Exception as e:
                        print(f"   ⚠️ Failed {method}-{device}: {str(e)}")
                        continue

                # Calculate composite score
                if param_results:
                    avg_silhouette = np.mean([r.silhouette_score for r in param_results])
                    avg_cev = np.mean([r.final_cev for r in param_results])
                    avg_time = np.mean([r.total_time_seconds for r in param_results])
                    avg_memory = np.mean([r.max_memory_mb for r in param_results])

                    # Composite score (higher is better)
                    composite_score = (
                        avg_silhouette * 0.4 +
                        avg_cev * 0.3 +
                        (1 / (1 + avg_time/60)) * 0.2 +  # Time penalty
                        (1 / (1 + avg_memory/1000)) * 0.1  # Memory penalty
                    )

                    grid_result = {
                        'level': level,
                        'kmer_size': kmer_size,
                        'params': params.copy(),
                        'composite_score': composite_score,
                        'avg_silhouette': avg_silhouette,
                        'avg_cev': avg_cev,
                        'avg_time': avg_time,
                        'avg_memory': avg_memory,
                        'n_successful': len(param_results)
                    }

                    grid_results.append(grid_result)

                    print(f"   ✅ Composite score: {composite_score:.4f}")

                    if composite_score > best_score:
                        best_score = composite_score
                        best_params = params.copy()

                # Cleanup
                del temp_benchmark
                gc.collect()

            except Exception as e:
                print(f"   ❌ Parameter set failed: {str(e)}")
                continue

        # Save grid search results
        grid_df = pd.DataFrame(grid_results)
        grid_filename = f"grid_search_{level}_kmer{kmer_size}_{self.timestamp}.csv"
        grid_df.to_csv(grid_filename, index=False)
        print(f"\n💾 Grid search results saved to: {grid_filename}")

        if best_params:
            print(f"🏆 Best parameters for {level}-kmer{kmer_size}:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")
            print(f"   Best composite score: {best_score:.4f}")

        return best_params, best_score, grid_results

    # =================================================================
    # ENHANCED BENCHMARKING
    # =================================================================

    # dimensionality_benchmark_production.py
    # Modifikasi run_single_benchmark_fold untuk menggunakan enhanced functionality

    # dimensionality_benchmark_production.py

    def run_single_benchmark_fold(self, X_train_sparse, y_train, method, device, level, kmer_size, fold):
        """Safe benchmark execution with proper error handling"""

        print(f"\n{'='*60}")
        print(f"🚀 PRODUCTION BENCHMARK: {method.upper()}-{device.upper()}")
        print(f"📊 Level: {level}, K-mer: {kmer_size}, Fold: {fold}")
        print(f"💾 RAM usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"📊 Data shape: {X_train_sparse.shape}")
        print(f"{'='*60}")


        # Enhanced device availability checks
        if device == 'gpu':
            if method in ['ipca', 'svd'] and not CUML_AVAILABLE:
                print(f"⚠️ {method.upper()} GPU not available, falling back to CPU")
                device = 'cpu'
            elif method == 'autoencoder' and not (TORCH_AVAILABLE and torch.cuda.is_available()):
                print(f"⚠️ Autoencoder GPU not available, falling back to CPU")
                device = 'cpu'
            elif method == 'umap' and not CUML_AVAILABLE:
                print(f"⚠️ UMAP GPU not available, falling back to CPU")
                device = 'cpu'

        monitor = MemoryMonitor()

        try:
            monitor.start_monitoring()
            total_start_time = time.time()

            # Use enhanced component finding
            search_start_time = time.time()

            if method in ['umap', 'autoencoder']:
                optimal_n, best_metric, cev_csv_path, cev_plot_path = self.find_optimal_components_manifold(
                    X_train_sparse, y_train, method, device, fold, level, kmer_size
                )
                optimization_method = "manifold"
                print(f"✅ Optimal components (manifold): {optimal_n} (score: {best_metric:.4f})")
            else:
                optimal_n, best_metric, cev_csv_path, cev_plot_path = self.find_optimal_components(
                    X_train_sparse, y_train, method, device, fold, level, kmer_size
                )
                optimization_method = "cev"
                print(f"✅ Optimal components (CEV): {optimal_n} (CEV: {best_metric*100:.2f}%)")

            search_time = time.time() - search_start_time

            # Transform data
            transform_start_time = time.time()
            X_transformed = self.transform_data(X_train_sparse, y_train, method, device, optimal_n)
            transform_time = time.time() - transform_start_time

            # Calculate silhouette score
            silhouette = self.calculate_silhouette_score(X_transformed, y_train)

            # Get manifold metrics if available
            manifold_metrics = getattr(self, 'manifold_metrics', {})

            # Save features
            features_csv_path = self.save_features_to_csv(
                X_transformed, y_train, method, device, fold, optimal_n, level, kmer_size
            )

            # Store features for plotting
            feature_key = f"{method}_{device}_{level}_kmer{kmer_size}_fold{fold}"
            self.production_features[feature_key] = {
                'features': X_transformed.copy(),
                'labels': y_train.copy(),
                'method': method,
                'device': device,
                'level': level,
                'kmer_size': kmer_size,
                'fold': fold,
                'n_components': optimal_n
            }

            total_time = time.time() - total_start_time
            monitor.stop_monitoring()
            cpu_max, cpu_avg, gpu_max, gpu_avg = monitor.get_stats()

            print(f"📊 Transform shape: {X_transformed.shape}")
            print(f"🎯 Silhouette score: {silhouette:.4f}")

            if optimization_method == "manifold" and manifold_metrics:
                print(f"🧮 Trustworthiness: {manifold_metrics.get('trustworthiness', 0.0):.4f}")
                print(f"🧮 Continuity: {manifold_metrics.get('continuity', 0.0):.4f}")
                print(f"🧮 Manifold Score: {manifold_metrics.get('combined_score', 0.0):.4f}")

            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"🧠 CPU Memory - Max: {cpu_max:.1f}MB, Avg: {cpu_avg:.1f}MB")
            if gpu_max > 0:
                print(f"🎮 GPU Memory - Max: {gpu_max:.1f}MB, Avg: {gpu_avg:.1f}MB")

            del X_transformed
            gc.collect()
            self.check_gpu_memory()

            # Create proper result
            result = BenchmarkResult(
                method=method,
                device=device,
                fold=fold,
                optimal_n=optimal_n,
                max_memory_mb=cpu_max,
                avg_memory_mb=cpu_avg,
                total_time_seconds=total_time,
                search_time_seconds=search_time,
                transform_time_seconds=transform_time,
                final_cev=best_metric if optimization_method == "cev" else 0.0,
                silhouette_score=silhouette,
                success=True,
                parameters={'level': level, 'kmer_size': kmer_size},
                cev_csv_path=cev_csv_path,
                cev_plot_path=cev_plot_path,
                features_csv_path=features_csv_path,
                timestamp=self.timestamp,
                trustworthiness_score=manifold_metrics.get('trustworthiness', 0.0),
                continuity_score=manifold_metrics.get('continuity', 0.0),
                procrustes_score=manifold_metrics.get('procrustes', 0.0),
                local_preservation_score=manifold_metrics.get('local_preservation', 0.0),
                intrinsic_dimension=manifold_metrics.get('intrinsic_dimension', 0.0),
                manifold_combined_score=manifold_metrics.get('combined_score', 0.0),
                optimization_method=optimization_method
            )

            return result

        except Exception as e:
            monitor.stop_monitoring()
            cpu_max, cpu_avg, gpu_max, gpu_avg = monitor.get_stats()

            print(f"❌ Error: {str(e)}")

            gc.collect()
            self.check_gpu_memory()

            return BenchmarkResult(
                method=method,
                device=device,
                fold=fold,
                optimal_n=0,
                max_memory_mb=cpu_max,
                avg_memory_mb=cpu_avg,
                total_time_seconds=0,
                search_time_seconds=0,
                transform_time_seconds=0,
                final_cev=0,
                silhouette_score=0,
                success=False,
                parameters={'level': level, 'kmer_size': kmer_size},
                timestamp=self.timestamp,
                error_message=str(e),
                optimization_method="failed"
            )
    
    def fit_model_for_test(self, X_train, y_train, method, device, n_components):
        """Fit model on training data and return fitted model for later use"""
        if method == 'svd' and device == 'cpu':
            Vt, _ = self.batch_svd_cpu(X_train, y_train, n_components)
            return {'type': 'svd_cpu', 'Vt': Vt}
            
        elif method == 'svd' and device == 'gpu':
            model, _ = self.svd_gpu(X_train, y_train, n_components)
            return {'type': 'svd_gpu', 'model': model}
            
        elif method == 'ipca' and device == 'cpu':
            model, _ = self.ipca_cpu(X_train, n_components)
            return {'type': 'ipca_cpu', 'model': model}
            
        elif method == 'ipca' and device == 'gpu':
            model, _ = self.ipca_gpu(X_train, n_components)
            return {'type': 'ipca_gpu', 'model': model}
            
        elif method == 'umap' and device == 'cpu':
            model, _ = self.umap_cpu(X_train, n_components)
            return {'type': 'umap_cpu', 'model': model}
            
        elif method == 'umap' and device == 'gpu':
            model, _ = self.umap_gpu(X_train, n_components)
            return {'type': 'umap_gpu', 'model': model}
            
        elif method == 'autoencoder' and device == 'cpu':
            model_dict, _ = self.autoencoder_cpu(X_train, n_components)
            return {'type': 'autoencoder_cpu', 'model_dict': model_dict}
            
        elif method == 'autoencoder' and device == 'gpu':
            model_dict, _ = self.autoencoder_gpu(X_train, n_components)
            return {'type': 'autoencoder_gpu', 'model_dict': model_dict}
        else:
            raise ValueError(f"Unknown method: {method}-{device}")
    
    def transform_with_fitted_model(self, X_data, fitted_model, method, device):
        """Transform data using pre-fitted model"""
        model_type = fitted_model['type']
        
        if model_type == 'svd_cpu':
            Vt = fitted_model['Vt']
            n_batches = int(np.ceil(X_data.shape[0] / self.batch_size))
            X_transformed = []
            
            for i in tqdm(range(n_batches), desc="SVD Transform Test (CPU)"):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, X_data.shape[0])
                batch = X_data[start_idx:end_idx].toarray()
                batch_transformed = batch @ Vt.T
                X_transformed.append(batch_transformed)
                del batch, batch_transformed
                if i % 3 == 0:
                    gc.collect()
            
            return np.vstack(X_transformed)
            
        elif model_type == 'svd_gpu':
            model = fitted_model['model']
            coo = X_data.tocoo()
            data_gpu = cp.asarray(coo.data, dtype=cp.float32)
            row_gpu = cp.asarray(coo.row, dtype=cp.int32)
            col_gpu = cp.asarray(coo.col, dtype=cp.int32)
            X_gpu = cp.sparse.csr_matrix((data_gpu, (row_gpu, col_gpu)),
                                       shape=coo.shape, dtype=cp.float32)
            X_transformed = model.transform(X_gpu).get()
            del X_gpu, data_gpu, row_gpu, col_gpu, coo
            self.check_gpu_memory()
            return X_transformed
            
        elif model_type == 'ipca_cpu':
            model = fitted_model['model']
            adaptive_batch = self.adaptive_batch_size(model.n_components, X_data.shape)
            X_transformed = []
            n_batches = int(np.ceil(X_data.shape[0] / adaptive_batch))

            for i in tqdm(range(n_batches), desc="IPCA Transform Test (CPU)"):
                start_idx = i * adaptive_batch
                end_idx = min((i + 1) * adaptive_batch, X_data.shape[0])
                batch = X_data[start_idx:end_idx].toarray()
                transformed = model.transform(batch)
                X_transformed.append(transformed)
                del batch, transformed
                if i % 3 == 0:
                    gc.collect()

            return np.vstack(X_transformed)
            
        elif model_type == 'ipca_gpu':
            model = fitted_model['model']
            if X_data.shape[0] > 5000:
                adaptive_batch = self.adaptive_batch_size(model.n_components, X_data.shape)
                X_transformed = []
                n_batches = int(np.ceil(X_data.shape[0] / adaptive_batch))

                for i in tqdm(range(n_batches), desc="IPCA Transform Test (GPU)"):
                    start_idx = i * adaptive_batch
                    end_idx = min((i + 1) * adaptive_batch, X_data.shape[0])
                    batch = X_data[start_idx:end_idx].toarray()
                    batch_gpu = cp.asarray(batch, dtype=cp.float32)
                    transformed = model.transform(batch_gpu).get()
                    X_transformed.append(transformed)
                    del batch, batch_gpu, transformed
                    if i % 3 == 0:
                        self.check_gpu_memory()
                        gc.collect()
                return np.vstack(X_transformed)
            else:
                X_dense = X_data.toarray()
                X_gpu = cp.asarray(X_dense, dtype=cp.float32)
                X_transformed = model.transform(X_gpu).get()
                del X_dense, X_gpu
                self.check_gpu_memory()
                return X_transformed
                
        elif model_type in ['umap_cpu', 'umap_gpu']:
            model = fitted_model['model']
            if model_type == 'umap_cpu':
                return model.transform(X_data.toarray())
            else:  # umap_gpu
                X_gpu = cp.asarray(X_data.toarray(), dtype=cp.float32)
                result = model.transform(X_gpu).get()
                del X_gpu
                self.check_gpu_memory()
                return result
                
        elif model_type in ['autoencoder_cpu', 'autoencoder_gpu']:
            model_dict = fitted_model['model_dict']
            X_dense = X_data.toarray().astype(np.float32)
            X_normalized = model_dict['scaler'].transform(X_dense)
            X_tensor = torch.FloatTensor(X_normalized).to(model_dict['device'])
            with torch.no_grad():
                result = model_dict['model'].encode(X_tensor).cpu().numpy()
            del X_dense, X_normalized, X_tensor
            if model_type == 'autoencoder_gpu':
                torch.cuda.empty_cache()
            return result
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_final_test_evaluation(self, X_train, y_train, X_test, y_test, method, device, level, kmer_size, optimal_n):
        """Final evaluation on held-out test set"""
        print(f"\n🧪 FINAL TEST EVALUATION: {method.upper()}-{device.upper()}")
        print(f"📊 Training on: {X_train.shape}, Testing on: {X_test.shape}")
        
        monitor = MemoryMonitor()
        
        try:
            monitor.start_monitoring()
            total_start_time = time.time()
            
            # 1. Fit on ALL training data
            print("🔧 Fitting final model on all training data...")
            final_model = self.fit_model_for_test(X_train, y_train, method, device, optimal_n)
            
            # 2. Transform test data
            print("🔧 Transforming test data...")
            X_test_transformed = self.transform_with_fitted_model(X_test, final_model, method, device)
            
            # 3. Calculate test metrics
            test_silhouette = self.calculate_silhouette_score(X_test_transformed, y_test)
            
            # Save test features
            features_csv_path = self.save_features_to_csv(
                X_test_transformed, y_test, method, device, -1, optimal_n, level, kmer_size, prefix="test"
            )
            
            # Store test features for plotting
            feature_key = f"{method}_{device}_{level}_kmer{kmer_size}_test"
            self.production_features[feature_key] = {
                'features': X_test_transformed.copy(),
                'labels': y_test.copy(),
                'method': method,
                'device': device,
                'level': level,
                'kmer_size': kmer_size,
                'fold': -1,  # Indicates test set
                'n_components': optimal_n,
                'data_type': 'test'
            }
            
            total_time = time.time() - total_start_time
            monitor.stop_monitoring()
            cpu_max, cpu_avg, gpu_max, gpu_avg = monitor.get_stats()
            
            print(f"📊 Test transform shape: {X_test_transformed.shape}")
            print(f"🎯 Test silhouette score: {test_silhouette:.4f}")
            print(f"⏱️  Test time: {total_time:.2f}s")
            
            # Clean up
            del X_test_transformed, final_model
            gc.collect()
            self.check_gpu_memory()
            
            # Create test result
            test_result = BenchmarkResult(
                method=method,
                device=device,
                fold=-1,  # Test set indicator
                optimal_n=optimal_n,
                max_memory_mb=cpu_max,
                avg_memory_mb=cpu_avg,
                total_time_seconds=total_time,
                search_time_seconds=0,  # No search on test
                transform_time_seconds=total_time,
                final_cev=0.0,  # CEV not applicable for test
                silhouette_score=test_silhouette,
                success=True,
                parameters={'level': level, 'kmer_size': kmer_size, 'data_type': 'test'},
                cev_csv_path="",
                cev_plot_path="",
                features_csv_path=features_csv_path,
                timestamp=self.timestamp,
                optimization_method="test"
            )
            
            return test_result
            
        except Exception as e:
            monitor.stop_monitoring()
            print(f"❌ Test evaluation failed: {str(e)}")
            return None
    
    def finalize_train_test_results(self, all_results):
        """Finalize results with train/test analysis"""
        # Save comprehensive results
        self.all_results_df = pd.DataFrame([r.to_dict() for r in all_results])
        results_filename = f"train_test_benchmark_results_{self.timestamp}.csv"
        self.all_results_df.to_csv(results_filename, index=False)
        print(f"\n💾 Train/Test results saved to: {results_filename}")

        # Generate comprehensive reports and plots
        self.generate_train_test_report()
        self.create_comprehensive_visualizations()
        self.plot_feature_spaces()

        return all_results
    
    def generate_train_test_report(self):
        """Generate train/test specific report"""
        if self.all_results_df is None:
            return

        print(f"\n{'='*80}")
        print(f"🏭 TRAIN/TEST BENCHMARK REPORT")
        print(f"{'='*80}")

        # Split by data type
        val_results = self.all_results_df[
            self.all_results_df['parameters'].str.contains('validation', na=False)
        ]
        test_results = self.all_results_df[
            self.all_results_df['parameters'].str.contains('test', na=False)
        ]

        print(f"📊 Validation Results: {len(val_results)} runs")
        print(f"📊 Test Results: {len(test_results)} runs")

        if not val_results.empty:
            successful_val = val_results[val_results['success'] == True]
            print(f"\n🎯 Validation Performance:")
            print(f"   - Success rate: {len(successful_val)/len(val_results):.1%}")
            print(f"   - Avg silhouette: {successful_val['silhouette_score'].mean():.4f}")
            print(f"   - Avg time: {successful_val['total_time_seconds'].mean():.2f}s")

        if not test_results.empty:
            successful_test = test_results[test_results['success'] == True]
            print(f"\n🧪 Test Performance:")
            print(f"   - Success rate: {len(successful_test)/len(test_results):.1%}")
            print(f"   - Avg silhouette: {successful_test['silhouette_score'].mean():.4f}")
            print(f"   - Avg time: {successful_test['total_time_seconds'].mean():.2f}s")

        # Compare validation vs test performance
        if not val_results.empty and not test_results.empty:
            print(f"\n📈 Validation vs Test Comparison:")
            val_avg = successful_val.groupby(['method', 'device'])['silhouette_score'].mean()
            test_avg = successful_test.groupby(['method', 'device'])['silhouette_score'].mean()
            
            for (method, device) in val_avg.index:
                if (method, device) in test_avg.index:
                    val_score = val_avg[(method, device)]
                    test_score = test_avg[(method, device)]
                    diff = test_score - val_score
                    print(f"   {method.upper()}-{device.upper()}: Val={val_score:.4f}, Test={test_score:.4f}, Diff={diff:+.4f}")

        # Save detailed report
        report_filename = f"train_test_report_detailed_{self.timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(f"Train/Test Benchmark Detailed Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write("DETAILED RESULTS\n")
            f.write(self.all_results_df.to_string(index=False))

        print(f"\n💾 Detailed train/test report saved to: {report_filename}")

    def save_features_to_csv(self, X_transformed, y, method, device, fold, n_components, level=None, kmer_size=None, prefix=""):
        """Enhanced save_features_to_csv with prefix support and output manager"""
        try:
            # Check if we have level and kmer_size for proper organization
            if level is not None and kmer_size is not None and hasattr(self, 'output_manager'):
                # Use output manager for organized output
                prefix_str = f"{prefix}_" if prefix else ""
                filename = f"{prefix_str}features_fold{fold}_{n_components}comp_{self.timestamp}.csv"
                
                # Create DataFrame
                feature_columns = [f'component_{i+1}' for i in range(X_transformed.shape[1])]
                df = pd.DataFrame(X_transformed, columns=feature_columns)
                df['label'] = y
                df['method'] = method
                df['device'] = device
                df['fold'] = fold
                df['n_components'] = n_components
                df['level'] = level
                df['kmer_size'] = kmer_size
                
                # Add metadata
                metadata = [
                    f"# Feature Extraction Results ({prefix.title() if prefix else 'Training'} Data)",
                    f"# Method: {method.upper()}",
                    f"# Device: {device.upper()}",
                    f"# Fold: {fold}",
                    f"# Level: {level}",
                    f"# K-mer Size: {kmer_size}",
                    f"# Components: {n_components}",
                    f"# Data Type: {prefix.title() if prefix else 'Training'}",
                    f"# Shape: {X_transformed.shape}",
                    f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"#"
                ]
                
                # Prepare content with metadata
                content_lines = metadata + ['']  # Add empty line after metadata
                content_lines.append(df.to_csv(index=False))
                content = '\n'.join(content_lines)
                
                # Use output manager
                file_path = self.output_manager.save_file_with_tracking(
                    content=content,
                    filename=filename,
                    level=level,
                    kmer=kmer_size,
                    method=method,
                    device=device,
                    file_type=f'features_{prefix}' if prefix else 'features',
                    save_func=lambda c, p: p.write_text(c, encoding='utf-8')
                )
                
                return str(file_path) if file_path else None
                
            else:
                # Fallback to old method for backward compatibility
                prefix_str = f"{prefix}_" if prefix else ""
                filename = f"{prefix_str}features_{method}_{device}_fold{fold}_{n_components}comp_{self.timestamp}.csv"
                
                # Create DataFrame
                feature_columns = [f'component_{i+1}' for i in range(X_transformed.shape[1])]
                df = pd.DataFrame(X_transformed, columns=feature_columns)
                df['label'] = y
                df['method'] = method
                df['device'] = device
                df['fold'] = fold
                df['n_components'] = n_components
                
                # Save to CSV
                df.to_csv(filename, index=False)
                print(f"💾 {prefix.title() if prefix else ''}Features saved to: {filename}")
                
                return filename
                
        except Exception as e:
            print(f"⚠️ Failed to save features: {str(e)}")
            return None
    
    def run_train_test_benchmark_fold(self, X_train, y_train, X_val, y_val, method, device, level, kmer_size, fold):
        """Train on X_train, validate on X_val - proper train/test split"""
        
        print(f"\n{'='*60}")
        print(f"🚀 TRAIN/TEST BENCHMARK: {method.upper()}-{device.upper()}")
        print(f"📊 Level: {level}, K-mer: {kmer_size}, Fold: {fold}")
        print(f"💾 RAM usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"📊 Train shape: {X_train.shape}, Val shape: {X_val.shape}")
        print(f"{'='*60}")

        # Enhanced device availability checks
        if device == 'gpu':
            if method in ['ipca', 'svd'] and not CUML_AVAILABLE:
                print(f"⚠️ {method.upper()} GPU not available, falling back to CPU")
                device = 'cpu'
            elif method == 'autoencoder' and not (TORCH_AVAILABLE and torch.cuda.is_available()):
                print(f"⚠️ Autoencoder GPU not available, falling back to CPU")
                device = 'cpu'
            elif method == 'umap' and not CUML_AVAILABLE:
                print(f"⚠️ UMAP GPU not available, falling back to CPU")
                device = 'cpu'

        monitor = MemoryMonitor()

        try:
            monitor.start_monitoring()
            total_start_time = time.time()

            # 1. Find optimal components using TRAINING data only
            search_start_time = time.time()

            if method in ['umap', 'autoencoder']:
                optimal_n, best_metric, cev_csv_path, cev_plot_path = self.find_optimal_components_manifold(
                    X_train, y_train, method, device, fold, level, kmer_size
                )
                optimization_method = "manifold"
                print(f"✅ Optimal components (manifold): {optimal_n} (score: {best_metric:.4f})")
            else:
                optimal_n, best_metric, cev_csv_path, cev_plot_path = self.find_optimal_components(
                    X_train, y_train, method, device, fold, level, kmer_size
                )
                optimization_method = "cev"
                print(f"✅ Optimal components (CEV): {optimal_n} (CEV: {best_metric*100:.2f}%)")

            search_time = time.time() - search_start_time

            # 2. Fit model on TRAINING data
            fit_start_time = time.time()
            trained_model = self.fit_model_for_test(X_train, y_train, method, device, optimal_n)
            fit_time = time.time() - fit_start_time

            # 2.5. Transform TRAINING data and save features
            train_transform_start_time = time.time()
            X_train_transformed = self.transform_with_fitted_model(X_train, trained_model, method, device)
            train_features_csv_path = self.save_features_to_csv(
                X_train_transformed, y_train, method, device, fold, optimal_n, level, kmer_size, prefix="train"
            )
            
            # Store train features for plotting
            train_feature_key = f"{method}_{device}_{level}_kmer{kmer_size}_fold{fold}_train"
            self.production_features[train_feature_key] = {
                'features': X_train_transformed.copy(),
                'labels': y_train.copy(),
                'method': method,
                'device': device,
                'level': level,
                'kmer_size': kmer_size,
                'fold': fold,
                'n_components': optimal_n,
                'data_type': 'train'
            }
            
            train_transform_time = time.time() - train_transform_start_time
            
            # Clean up train transformed data
            del X_train_transformed
            gc.collect()

            # 3. Transform VALIDATION data using trained model
            transform_start_time = time.time()
            X_val_transformed = self.transform_with_fitted_model(X_val, trained_model, method, device)
            transform_time = time.time() - transform_start_time

            # 4. Calculate validation metrics
            val_silhouette = self.calculate_silhouette_score(X_val_transformed, y_val)

            # Get manifold metrics if available
            manifold_metrics = getattr(self, 'manifold_metrics', {})

            # Save validation features
            features_csv_path = self.save_features_to_csv(
                X_val_transformed, y_val, method, device, fold, optimal_n, level, kmer_size, prefix="val"
            )

            # Store validation features for plotting
            feature_key = f"{method}_{device}_{level}_kmer{kmer_size}_fold{fold}_val"
            self.production_features[feature_key] = {
                'features': X_val_transformed.copy(),
                'labels': y_val.copy(),
                'method': method,
                'device': device,
                'level': level,
                'kmer_size': kmer_size,
                'fold': fold,
                'n_components': optimal_n,
                'data_type': 'validation'
            }

            total_time = time.time() - total_start_time
            monitor.stop_monitoring()
            cpu_max, cpu_avg, gpu_max, gpu_avg = monitor.get_stats()

            print(f"📊 Val transform shape: {X_val_transformed.shape}")
            print(f"🎯 Validation silhouette score: {val_silhouette:.4f}")

            if optimization_method == "manifold" and manifold_metrics:
                print(f"🧮 Trustworthiness: {manifold_metrics.get('trustworthiness', 0.0):.4f}")
                print(f"🧮 Continuity: {manifold_metrics.get('continuity', 0.0):.4f}")
                print(f"🧮 Manifold Score: {manifold_metrics.get('combined_score', 0.0):.4f}")

            print(f"⏱️  Total time: {total_time:.2f}s (fit: {fit_time:.2f}s, train_transform: {train_transform_time:.2f}s, val_transform: {transform_time:.2f}s)")
            print(f"🧠 CPU Memory - Max: {cpu_max:.1f}MB, Avg: {cpu_avg:.1f}MB")
            if gpu_max > 0:
                print(f"🎮 GPU Memory - Max: {gpu_max:.1f}MB, Avg: {gpu_avg:.1f}MB")

            # Clean up
            del X_val_transformed, trained_model
            gc.collect()
            self.check_gpu_memory()

            # Create proper result
            result = BenchmarkResult(
                method=method,
                device=device,
                fold=fold,
                optimal_n=optimal_n,
                max_memory_mb=cpu_max,
                avg_memory_mb=cpu_avg,
                total_time_seconds=total_time,
                search_time_seconds=search_time,
                transform_time_seconds=transform_time,
                final_cev=best_metric if optimization_method == "cev" else 0.0,
                silhouette_score=val_silhouette,  # This is validation score!
                success=True,
                parameters={'level': level, 'kmer_size': kmer_size, 'data_type': 'validation', 'train_features_saved': True},
                cev_csv_path=cev_csv_path,
                cev_plot_path=cev_plot_path,
                features_csv_path=features_csv_path,  # This is validation features path
                timestamp=self.timestamp,
                trustworthiness_score=manifold_metrics.get('trustworthiness', 0.0),
                continuity_score=manifold_metrics.get('continuity', 0.0),
                procrustes_score=manifold_metrics.get('procrustes', 0.0),
                local_preservation_score=manifold_metrics.get('local_preservation', 0.0),
                intrinsic_dimension=manifold_metrics.get('intrinsic_dimension', 0.0),
                manifold_combined_score=manifold_metrics.get('combined_score', 0.0),
                optimization_method=optimization_method
            )

            return result

        except Exception as e:
            monitor.stop_monitoring()
            cpu_max, cpu_avg, gpu_max, gpu_avg = monitor.get_stats()

            print(f"❌ Error: {str(e)}")

            gc.collect()
            self.check_gpu_memory()

            return BenchmarkResult(
                method=method,
                device=device,
                fold=fold,
                optimal_n=0,
                max_memory_mb=cpu_max,
                avg_memory_mb=cpu_avg,
                total_time_seconds=0,
                search_time_seconds=0,
                transform_time_seconds=0,
                final_cev=0,
                silhouette_score=0,
                success=False,
                parameters={'level': level, 'kmer_size': kmer_size, 'data_type': 'validation'},
                timestamp=self.timestamp,
                error_message=str(e),
            )

    def run_production_benchmark_with_test(self, train_data_dict, test_data_dict=None):
        """
        Run comprehensive production benchmark with proper train/test split
        
        Parameters:
        train_data_dict: {(level, kmer_size): (X_train_sparse, y_train)}
        test_data_dict: {(level, kmer_size): (X_test_sparse, y_test)} (optional)
        """
        print(f"\n🏭 STARTING PRODUCTION BENCHMARK WITH TRAIN/TEST SPLIT")
        print(f"📊 Total combinations: {len(train_data_dict)}")
        print(f"💾 Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
        
        if test_data_dict:
            print(f"✅ Test data provided - will evaluate on held-out test set")
        else:
            print(f"⚠️  No test data - using cross-validation only")

        all_results = []
        combinations = self.get_available_combinations()

        total_benchmarks = len(train_data_dict) * len(combinations) * self.cv_folds
        current_benchmark = 0
        
        # Store best models for each combination
        best_models = {}

        for (level, kmer_size), (X_train_sparse, y_train) in train_data_dict.items():
            print(f"\n{'='*80}")
            print(f"🧬 PROCESSING: {level.upper()} - K-mer {kmer_size}")
            print(f"📊 Train data shape: {X_train_sparse.shape}")
            print(f"🏷️  Unique labels: {len(np.unique(y_train))}")
            
            # Check if test data exists for this combination
            test_available = test_data_dict and (level, kmer_size) in test_data_dict
            if test_available:
                X_test_sparse, y_test = test_data_dict[(level, kmer_size)]
                print(f"📊 Test data shape: {X_test_sparse.shape}")
            
            print(f"{'='*80}")

            # Cross-validation on training data
            if self.cv_folds > 1 and len(np.unique(y_train)) > 1:
                try:
                    skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                    folds = list(skf.split(X_train_sparse, y_train))
                except:
                    # Fallback to regular KFold
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                    folds = list(kf.split(X_train_sparse))
            else:
                # Single fold
                folds = [(np.arange(len(y_train)), np.arange(len(y_train)))]

            # Track best performance per method for this level-kmer combination
            method_performance = {}

            for fold, (train_idx, val_idx) in enumerate(folds):
                X_fold_train = X_train_sparse[train_idx] if len(train_idx) < len(y_train) else X_train_sparse
                y_fold_train = y_train[train_idx] if len(train_idx) < len(y_train) else y_train
                
                # Validation set from training data
                X_fold_val = X_train_sparse[val_idx] if len(val_idx) < len(y_train) else X_train_sparse
                y_fold_val = y_train[val_idx] if len(val_idx) < len(y_train) else y_train

                print(f"\n🔄 Fold {fold + 1}/{len(folds)}")
                print(f"📊 Fold train shape: {X_fold_train.shape}")
                print(f"📊 Fold val shape: {X_fold_val.shape}")

                for method, device in combinations:
                    current_benchmark += 1
                    print(f"\n🎯 Progress: {current_benchmark}/{total_benchmarks}")

                    # Memory check
                    if psutil.virtual_memory().percent > 75:
                        print("⚠️  High memory usage, cleaning up...")
                        gc.collect()
                        self.check_gpu_memory()
                        time.sleep(3)

                    # Train and validate
                    result = self.run_train_test_benchmark_fold(
                        X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                        method, device, level, kmer_size, fold
                    )
                    all_results.append(result)
                    
                    # Track best model for each method
                    method_key = f"{method}_{device}"
                    if method_key not in method_performance:
                        method_performance[method_key] = {'best_score': -1, 'best_result': None}
                    
                    if result.success and result.silhouette_score > method_performance[method_key]['best_score']:
                        method_performance[method_key]['best_score'] = result.silhouette_score
                        method_performance[method_key]['best_result'] = result

                    gc.collect()
                    self.check_gpu_memory()
                    time.sleep(2)
            
            # Test on held-out test set with best models
            if test_available:
                print(f"\n🧪 TESTING ON HELD-OUT TEST SET")
                print(f"📊 Test data shape: {X_test_sparse.shape}")
                
                for method_key, perf_data in method_performance.items():
                    if perf_data['best_result'] and perf_data['best_result'].success:
                        method, device = method_key.split('_')
                        best_result = perf_data['best_result']
                        
                        print(f"\n🔬 Testing {method.upper()}-{device.upper()} (best CV model)")
                        
                        test_result = self.run_final_test_evaluation(
                            X_train_sparse, y_train, X_test_sparse, y_test,
                            method, device, level, kmer_size, best_result.optimal_n
                        )
                        
                        if test_result:
                            all_results.append(test_result)

        return self.finalize_train_test_results(all_results)
    
    def  run_production_benchmark(self, data_dict):
        """
        Run comprehensive production benchmark (legacy method)
        data_dict format: {(level, kmer_size): (X_sparse, y)}
        """
        print(f"\n🏭 STARTING PRODUCTION BENCHMARK")
        print(f"📊 Total combinations: {len(data_dict)}")
        print(f"💾 Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")

        all_results = []
        combinations = self.get_available_combinations()

        total_benchmarks = len(data_dict) * len(combinations) * self.cv_folds
        current_benchmark = 0

        for (level, kmer_size), (X_sparse, y) in data_dict.items():
            print(f"\n{'='*80}")
            print(f"🧬 PROCESSING: {level.upper()} - K-mer {kmer_size}")
            print(f"📊 Data shape: {X_sparse.shape}")
            print(f"🏷️  Unique labels: {len(np.unique(y))}")
            print(f"{'='*80}")

            # Cross-validation
            if self.cv_folds > 1 and len(np.unique(y)) > 1:
                try:
                    skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                    folds = list(skf.split(X_sparse, y))
                except:
                    # Fallback to regular KFold
                    kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                    folds = list(kf.split(X_sparse))
            else:
                # Single fold
                folds = [(np.arange(len(y)), np.arange(len(y)))]

            for fold, (train_idx, val_idx) in enumerate(folds):
                X_fold = X_sparse[train_idx] if len(train_idx) < len(y) else X_sparse
                y_fold = y[train_idx] if len(train_idx) < len(y) else y

                print(f"\n🔄 Fold {fold + 1}/{len(folds)}")
                print(f"📊 Fold data shape: {X_fold.shape}")

                for method, device in combinations:
                    current_benchmark += 1
                    print(f"\n🎯 Progress: {current_benchmark}/{total_benchmarks}")

                    # Memory check
                    if psutil.virtual_memory().percent > 75:
                        print("⚠️  High memory usage, cleaning up...")
                        gc.collect()
                        self.check_gpu_memory()
                        time.sleep(3)

                    result = self.run_single_benchmark_fold(
                        X_fold, y_fold, method, device, level, kmer_size, fold
                    )
                    all_results.append(result)

                    gc.collect()
                    self.check_gpu_memory()
                    time.sleep(2)

        # Save comprehensive results
        self.all_results_df = pd.DataFrame([r.to_dict() for r in all_results])
        results_filename = f"production_benchmark_results_{self.timestamp}.csv"
        self.all_results_df.to_csv(results_filename, index=False)
        print(f"\n💾 All results saved to: {results_filename}")

        # Generate comprehensive reports and plots
        self.generate_production_report()
        self.create_comprehensive_visualizations()
        self.plot_feature_spaces()

        return all_results

    # =================================================================
    # COMPREHENSIVE VISUALIZATION
    # =================================================================

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualization suite"""
        if self.all_results_df is None or self.all_results_df.empty:
            print("⚠️  No results to visualize")
            return

        successful_df = self.all_results_df[self.all_results_df['success'] == True]

        if successful_df.empty:
            print("⚠️  No successful results to visualize")
            return

        # Extract level and kmer_size from parameters
        def extract_params(param_str):
            try:
                params = eval(param_str)
                return params.get('level', 'unknown'), params.get('kmer_size', 0)
            except:
                return 'unknown', 0

        successful_df[['level', 'kmer_size']] = successful_df['parameters'].apply(
            lambda x: pd.Series(extract_params(x))
        )

        print(f"📊 Creating comprehensive visualizations...")

        # 1. Success Rate Analysis
        self.plot_success_rate_analysis(successful_df)

        # 2. Performance Distributions
        self.plot_performance_distributions(successful_df)

        # 3. Method Comparison Heatmaps
        self.plot_method_comparison_heatmaps(successful_df)

        # 4. CEV vs Silhouette Analysis
        self.plot_cev_silhouette_analysis(successful_df)

        # 5. Production Recommendations
        self.plot_production_recommendations(successful_df)

    def plot_success_rate_analysis(self, df):
        """Plot success rate analysis with GridSpec for better spacing control"""
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25,
                    top=0.92, bottom=0.08, left=0.06, right=0.94)

        fig.suptitle('Success Rate Analysis', fontsize=18, fontweight='bold')

        # 1. Success rate by method-device
        ax1 = fig.add_subplot(gs[0, 0])
        method_device = df.groupby(['method', 'device'])['success'].agg(['count', 'sum']).reset_index()
        method_device['success_rate'] = method_device['sum'] / method_device['count']
        method_device['label'] = method_device['method'] + '-' + method_device['device']

        bars1 = ax1.bar(method_device['label'], method_device['success_rate'],
                        color=plt.cm.Set3(np.arange(len(method_device))))

        ax1.set_title('Success Rate by Method-Device', fontsize=12, fontweight='bold', pad=15)
        ax1.set_ylabel('Success Rate', fontsize=11)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.set_ylim(0, 1.1)

        for bar, rate in zip(bars1, method_device['success_rate']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 2. Success rate by taxonomic level
        ax2 = fig.add_subplot(gs[0, 1])
        level_success = df.groupby('level')['success'].agg(['count', 'sum']).reset_index()
        level_success['success_rate'] = level_success['sum'] / level_success['count']

        bars2 = ax2.bar(level_success['level'], level_success['success_rate'],
                        color=plt.cm.Set2(np.arange(len(level_success))))

        ax2.set_title('Success Rate by Taxonomic Level', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylabel('Success Rate', fontsize=11)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.set_ylim(0, 1.1)

        for bar, rate in zip(bars2, level_success['success_rate']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 3. Success rate by k-mer size
        ax3 = fig.add_subplot(gs[1, 0])
        kmer_success = df.groupby('kmer_size')['success'].agg(['count', 'sum']).reset_index()
        kmer_success['success_rate'] = kmer_success['sum'] / kmer_success['count']

        bars3 = ax3.bar([f'K-mer {k}' for k in kmer_success['kmer_size']],
                        kmer_success['success_rate'],
                        color=plt.cm.Set1(np.arange(len(kmer_success))))

        ax3.set_title('Success Rate by K-mer Size', fontsize=12, fontweight='bold', pad=15)
        ax3.set_ylabel('Success Rate', fontsize=11)
        ax3.tick_params(axis='x', labelsize=9)
        ax3.set_ylim(0, 1.1)

        for bar, rate in zip(bars3, kmer_success['success_rate']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 4. Combined heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        pivot_data = df.pivot_table(values='success', index='level',
                                columns='kmer_size', aggfunc='mean', fill_value=0)

        im = ax4.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax4.set_title('Success Rate Heatmap\n(Level vs K-mer)', fontsize=12, fontweight='bold', pad=15)
        ax4.set_xticks(range(len(pivot_data.columns)))
        ax4.set_xticklabels([f'K-mer {k}' for k in pivot_data.columns], fontsize=9)
        ax4.set_yticks(range(len(pivot_data.index)))
        ax4.set_yticklabels(pivot_data.index, fontsize=9)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
        cbar.set_label('Success Rate', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Add percentage annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                ax4.text(j, i, f'{pivot_data.iloc[i, j]:.1%}',
                        ha="center", va="center", color="black",
                        fontweight='bold', fontsize=9)

        filename = f"success_rate_analysis_fixed_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Success rate analysis (fixed layout) saved to: {filename}")

    def plot_success_rate_analysis_1(self, df):
        """Plot success rate analysis across different dimensions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Success Rate Analysis', fontsize=16, fontweight='bold')

        # 1. Success rate by method-device
        method_device = df.groupby(['method', 'device'])['success'].agg(['count', 'sum']).reset_index()
        method_device['success_rate'] = method_device['sum'] / method_device['count']
        method_device['label'] = method_device['method'] + '-' + method_device['device']

        bars1 = axes[0,0].bar(method_device['label'], method_device['success_rate'],
                             color=plt.cm.Set3(np.arange(len(method_device))))
        axes[0,0].set_title('Success Rate by Method-Device')
        axes[0,0].set_ylabel('Success Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].set_ylim(0, 1)

        # Add percentage labels
        for bar, rate in zip(bars1, method_device['success_rate']):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom')

        # 2. Success rate by taxonomic level
        level_success = df.groupby('level')['success'].agg(['count', 'sum']).reset_index()
        level_success['success_rate'] = level_success['sum'] / level_success['count']

        bars2 = axes[0,1].bar(level_success['level'], level_success['success_rate'],
                             color=plt.cm.Set2(np.arange(len(level_success))))
        axes[0,1].set_title('Success Rate by Taxonomic Level')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].set_ylim(0, 1)

        for bar, rate in zip(bars2, level_success['success_rate']):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom')

        # 3. Success rate by k-mer size
        kmer_success = df.groupby('kmer_size')['success'].agg(['count', 'sum']).reset_index()
        kmer_success['success_rate'] = kmer_success['sum'] / kmer_success['count']

        bars3 = axes[1,0].bar([f'K-mer {k}' for k in kmer_success['kmer_size']],
                             kmer_success['success_rate'],
                             color=plt.cm.Set1(np.arange(len(kmer_success))))
        axes[1,0].set_title('Success Rate by K-mer Size')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_ylim(0, 1)

        for bar, rate in zip(bars3, kmer_success['success_rate']):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{rate:.1%}', ha='center', va='bottom')

        # 4. Combined heatmap
        pivot_data = df.pivot_table(values='success', index='level',
                                   columns='kmer_size', aggfunc='mean', fill_value=0)

        im = axes[1,1].imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1,1].set_title('Success Rate Heatmap\n(Level vs K-mer)')
        axes[1,1].set_xticks(range(len(pivot_data.columns)))
        axes[1,1].set_xticklabels([f'K-mer {k}' for k in pivot_data.columns])
        axes[1,1].set_yticks(range(len(pivot_data.index)))
        axes[1,1].set_yticklabels(pivot_data.index)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1,1])
        cbar.set_label('Success Rate')

        # Add percentage annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = axes[1,1].text(j, i, f'{pivot_data.iloc[i, j]:.1%}',
                                     ha="center", va="center", color="black", fontweight='bold')

        plt.tight_layout()

        filename = f"success_rate_analysis_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Success rate analysis saved to: {filename}")

    def plot_performance_distributions(self, df):
        """Plot performance distributions with fixed color issues"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')

        # Create consistent color mapping
        methods = df['method'].unique()
        devices = df['device'].unique()
        method_colors = {method: plt.cm.Set1(i) for i, method in enumerate(methods)}
        device_shapes = {'cpu': 'o', 'gpu': 's'}

        # 1. Execution Time Distribution
        method_device_groups = df.groupby(['method', 'device'])

        box_data = []
        box_labels = []
        box_colors = []

        for (method, device), group in method_device_groups:
            if not group.empty and group['total_time_seconds'].notna().any():
                box_data.append(group['total_time_seconds'].dropna())
                box_labels.append(f'{method.upper()}-{device.upper()}')
                box_colors.append(method_colors[method])

        if box_data:
            bp1 = axes[0,0].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp1['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[0,0].set_title('Execution Time Distribution')
            axes[0,0].set_ylabel('Time (seconds)')
            axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Cumulative Explained Variance Distribution
        box_data = []
        box_labels = []
        box_colors = []

        for (method, device), group in method_device_groups:
            if not group.empty and group['final_cev'].notna().any():
                box_data.append(group['final_cev'].dropna() * 100)
                box_labels.append(f'{method.upper()}-{device.upper()}')
                box_colors.append(method_colors[method])

        if box_data:
            bp2 = axes[0,1].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp2['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[0,1].set_title('Cumulative Explained Variance Distribution')
            axes[0,1].set_ylabel('CEV (%)')
            axes[0,1].tick_params(axis='x', rotation=45)

        # 3. Silhouette Score Distribution
        box_data = []
        box_labels = []
        box_colors = []

        for (method, device), group in method_device_groups:
            if not group.empty and group['silhouette_score'].notna().any():
                box_data.append(group['silhouette_score'].dropna())
                box_labels.append(f'{method.upper()}-{device.upper()}')
                box_colors.append(method_colors[method])

        if box_data:
            bp3 = axes[1,0].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp3['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[1,0].set_title('Silhouette Score Distribution')
            axes[1,0].set_ylabel('Silhouette Score')
            axes[1,0].tick_params(axis='x', rotation=45)

        # 4. Peak Memory Distribution
        box_data = []
        box_labels = []
        box_colors = []

        for (method, device), group in method_device_groups:
            if not group.empty and group['max_memory_mb'].notna().any():
                box_data.append(group['max_memory_mb'].dropna())
                box_labels.append(f'{method.upper()}-{device.upper()}')
                box_colors.append(method_colors[method])

        if box_data:
            bp4 = axes[1,1].boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp4['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            axes[1,1].set_title('Peak Memory Distribution')
            axes[1,1].set_ylabel('Memory (MB)')
            axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        filename = f"performance_distributions_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Performance distributions saved to: {filename}")

    def plot_method_comparison_heatmaps(self, df):
        """Create method comparison heatmaps"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Method Comparison Heatmaps', fontsize=16, fontweight='bold')

        # Create method-device labels
        df['method_device'] = df['method'] + '-' + df['device']

        # 1. Average Silhouette Score
        pivot1 = df.pivot_table(values='silhouette_score', index='level',
                               columns='method_device', aggfunc='mean', fill_value=0)

        im1 = axes[0,0].imshow(pivot1.values, cmap='viridis', aspect='auto')
        axes[0,0].set_title('Average Silhouette Score')
        axes[0,0].set_xticks(range(len(pivot1.columns)))
        axes[0,0].set_xticklabels(pivot1.columns, rotation=45, ha='right')
        axes[0,0].set_yticks(range(len(pivot1.index)))
        axes[0,0].set_yticklabels(pivot1.index)

        # Add annotations
        for i in range(len(pivot1.index)):
            for j in range(len(pivot1.columns)):
                axes[0,0].text(j, i, f'{pivot1.iloc[i, j]:.3f}',
                              ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im1, ax=axes[0,0])

        # 2. Average CEV
        pivot2 = df.pivot_table(values='final_cev', index='level',
                               columns='method_device', aggfunc='mean', fill_value=0)

        im2 = axes[0,1].imshow(pivot2.values, cmap='plasma', aspect='auto')
        axes[0,1].set_title('Average CEV')
        axes[0,1].set_xticks(range(len(pivot2.columns)))
        axes[0,1].set_xticklabels(pivot2.columns, rotation=45, ha='right')
        axes[0,1].set_yticks(range(len(pivot2.index)))
        axes[0,1].set_yticklabels(pivot2.index)

        for i in range(len(pivot2.index)):
            for j in range(len(pivot2.columns)):
                axes[0,1].text(j, i, f'{pivot2.iloc[i, j]:.3f}',
                              ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im2, ax=axes[0,1])

        # 3. Average Execution Time
        pivot3 = df.pivot_table(values='total_time_seconds', index='level',
                               columns='method_device', aggfunc='mean', fill_value=0)

        im3 = axes[1,0].imshow(pivot3.values, cmap='YlOrRd', aspect='auto')
        axes[1,0].set_title('Average Execution Time (s)')
        axes[1,0].set_xticks(range(len(pivot3.columns)))
        axes[1,0].set_xticklabels(pivot3.columns, rotation=45, ha='right')
        axes[1,0].set_yticks(range(len(pivot3.index)))
        axes[1,0].set_yticklabels(pivot3.index)

        for i in range(len(pivot3.index)):
            for j in range(len(pivot3.columns)):
                axes[1,0].text(j, i, f'{pivot3.iloc[i, j]:.1f}',
                              ha="center", va="center", color="white", fontsize=8)

        plt.colorbar(im3, ax=axes[1,0])

        # 4. Average Memory Usage
        pivot4 = df.pivot_table(values='max_memory_mb', index='level',
                               columns='method_device', aggfunc='mean', fill_value=0)

        im4 = axes[1,1].imshow(pivot4.values, cmap='coolwarm', aspect='auto')
        axes[1,1].set_title('Average Peak Memory (MB)')
        axes[1,1].set_xticks(range(len(pivot4.columns)))
        axes[1,1].set_xticklabels(pivot4.columns, rotation=45, ha='right')
        axes[1,1].set_yticks(range(len(pivot4.index)))
        axes[1,1].set_yticklabels(pivot4.index)

        for i in range(len(pivot4.index)):
            for j in range(len(pivot4.columns)):
                axes[1,1].text(j, i, f'{pivot4.iloc[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im4, ax=axes[1,1])

        plt.tight_layout()

        filename = f"method_comparison_heatmaps_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Method comparison heatmaps saved to: {filename}")

    def plot_cev_silhouette_analysis(self, df):
        """Plot CEV vs Silhouette analysis for each combination/fold"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CEV vs Silhouette Score Analysis', fontsize=16, fontweight='bold')

        # Color mapping
        methods = df['method'].unique()
        method_colors = {method: plt.cm.Set1(i) for i, method in enumerate(methods)}

        # 1. CEV vs Silhouette Score scatter plot
        for method in methods:
            method_data = df[df['method'] == method]
            if not method_data.empty:
                axes[0,0].scatter(method_data['final_cev'] * 100,
                                 method_data['silhouette_score'],
                                 c=method_colors[method], label=method.upper(),
                                 alpha=0.7, s=50)

        axes[0,0].set_xlabel('CEV (%)')
        axes[0,0].set_ylabel('Silhouette Score')
        axes[0,0].set_title('CEV vs Silhouette Score by Method')
        axes[0,0].legend()
        # axes[0,0].grid(True, alpha=0.3)

        # 2. By device
        devices = df['device'].unique()
        device_markers = {'cpu': 'o', 'gpu': 's'}

        for device in devices:
            device_data = df[df['device'] == device]
            if not device_data.empty:
                axes[0,1].scatter(device_data['final_cev'] * 100,
                                 device_data['silhouette_score'],
                                 marker=device_markers.get(device, 'o'),
                                 label=device.upper(), alpha=0.7, s=50)

        axes[0,1].set_xlabel('CEV (%)')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].set_title('CEV vs Silhouette Score by Device')
        axes[0,1].legend()
        # axes[0,1].grid(True, alpha=0.3)

        # 3. By taxonomic level
        levels = df['level'].unique()
        level_colors = {level: plt.cm.Set2(i) for i, level in enumerate(levels)}

        for level in levels:
            level_data = df[df['level'] == level]
            if not level_data.empty:
                axes[1,0].scatter(level_data['final_cev'] * 100,
                                 level_data['silhouette_score'],
                                 c=level_colors[level], label=level.upper(),
                                 alpha=0.7, s=50)

        axes[1,0].set_xlabel('CEV (%)')
        axes[1,0].set_ylabel('Silhouette Score')
        axes[1,0].set_title('CEV vs Silhouette Score by Taxonomic Level')
        axes[1,0].legend()
        # axes[1,0].grid(True, alpha=0.3)

        # 4. By k-mer size
        kmer_sizes = sorted(df['kmer_size'].unique())
        kmer_colors = {kmer: plt.cm.Set3(i) for i, kmer in enumerate(kmer_sizes)}

        for kmer in kmer_sizes:
            kmer_data = df[df['kmer_size'] == kmer]
            if not kmer_data.empty:
                axes[1,1].scatter(kmer_data['final_cev'] * 100,
                                 kmer_data['silhouette_score'],
                                 c=kmer_colors[kmer], label=f'K-mer {kmer}',
                                 alpha=0.7, s=50)

        axes[1,1].set_xlabel('CEV (%)')
        axes[1,1].set_ylabel('Silhouette Score')
        axes[1,1].set_title('CEV vs Silhouette Score by K-mer Size')
        axes[1,1].legend()
        # axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"cev_silhouette_analysis_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 CEV vs Silhouette analysis saved to: {filename}")

    def plot_feature_spaces(self):
        """Plot feature spaces with compact colorbar legend"""
        if not self.production_features:
            print("⚠️  No features available for plotting")
            return

        methods_to_plot = ['ipca', 'svd', 'umap', 'autoencoder']
        available_features = {}

        # Group features by method
        for key, feature_data in self.production_features.items():
            method = feature_data['method']
            if method in methods_to_plot:
                if method not in available_features:
                    available_features[method] = []
                available_features[method].append((key, feature_data))

        if not available_features:
            print("⚠️  No target methods available for feature space plotting")
            return

        # Create subplots for each method
        n_methods = len(available_features)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        fig.suptitle('Feature Space Visualization with Colorbar Legend', fontsize=16, fontweight='bold')

        method_idx = 0
        for method, feature_list in available_features.items():
            if method_idx >= 4:  # Maximum 4 subplots
                break

            ax = axes[method_idx]

            # Plot first available feature set for this method
            if feature_list:
                key, feature_data = feature_list[0]  # Take first available
                features = feature_data['features']
                labels = feature_data['labels']
                level = feature_data['level']
                kmer_size = feature_data['kmer_size']

                # Use first 2 components for visualization
                if features.shape[1] >= 2:
                    x = features[:, 0]
                    y = features[:, 1]
                else:
                    # For 1D case, use component vs index
                    x = np.arange(len(features))
                    y = features[:, 0]

                # Create label mapping for colorbar
                unique_labels = np.unique(labels)
                n_labels = len(unique_labels)
                label_to_num = {label: i for i, label in enumerate(unique_labels)}
                colors_numeric = np.array([label_to_num[label] for label in labels])

                # Choose colormap
                if n_labels <= 20:
                    cmap = plt.cm.tab20
                else:
                    cmap = plt.cm.viridis

                # Create scatter plot
                scatter = ax.scatter(x, y, c=colors_numeric, cmap=cmap,
                                alpha=0.7, s=20, vmin=0, vmax=max(1, n_labels-1))

                ax.set_title(f'{method.upper()}\n{level} - K-mer {kmer_size} ({n_labels} labels)')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                # ax.grid(True, alpha=0.3)

                # Add compact colorbar
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)

                if n_labels <= 10:
                    # Show all labels
                    cbar.set_ticks(np.arange(n_labels))
                    cbar.set_ticklabels([f'{i}' for i in range(n_labels)])
                    cbar.set_label('Label ID', fontsize=8)
                else:
                    # Show subset
                    step = max(1, n_labels // 8)
                    tick_positions = np.arange(0, n_labels, step)
                    cbar.set_ticks(tick_positions)
                    cbar.set_ticklabels([f'{i}' for i in tick_positions])
                    cbar.set_label(f'Label ID (0-{n_labels-1})', fontsize=8)

                cbar.ax.tick_params(labelsize=6)

            method_idx += 1

        # Hide unused subplots
        for i in range(method_idx, 4):
            axes[i].set_visible(False)

        plt.tight_layout()

        filename = f"feature_spaces_compact_colorbar_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Feature spaces with compact colorbar saved to: {filename}")

    def plot_feature_spaces_1(self):
        """Plot feature spaces for UMAP, PCA, SVD, Autoencoder"""
        if not self.production_features:
            print("⚠️  No features available for plotting")
            return

        methods_to_plot = ['ipca', 'svd', 'umap', 'autoencoder']
        available_features = {}

        # Group features by method
        for key, feature_data in self.production_features.items():
            method = feature_data['method']
            if method in methods_to_plot:
                if method not in available_features:
                    available_features[method] = []
                available_features[method].append((key, feature_data))

        if not available_features:
            print("⚠️  No target methods available for feature space plotting")
            return

        # Create subplots for each method
        n_methods = len(available_features)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        fig.suptitle('Feature Space Visualization (First 2 Components)', fontsize=16, fontweight='bold')

        method_idx = 0
        for method, feature_list in available_features.items():
            if method_idx >= 4:  # Maximum 4 subplots
                break

            ax = axes[method_idx]

            # Plot first available feature set for this method
            if feature_list:
                key, feature_data = feature_list[0]  # Take first available
                features = feature_data['features']
                labels = feature_data['labels']
                level = feature_data['level']
                kmer_size = feature_data['kmer_size']

                # Use first 2 components for visualization
                if features.shape[1] >= 2:
                    x = features[:, 0]
                    y = features[:, 1]
                else:
                    # For 1D case, use component vs index
                    x = np.arange(len(features))
                    y = features[:, 0]

                # Create scatter plot
                unique_labels = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(x[mask], y[mask], c=[colors[i]],
                              label=str(label), alpha=0.7, s=20)

                ax.set_title(f'{method.upper()}\n{level} - K-mer {kmer_size}')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)

            method_idx += 1

        # Hide unused subplots
        for i in range(method_idx, 4):
            axes[i].set_visible(False)

        plt.tight_layout()

        filename = f"feature_spaces_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Feature spaces saved to: {filename}")

    def plot_production_recommendations(self, df):
        """Generate production recommendations visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Production Recommendations', fontsize=16, fontweight='bold')

        # 1. Best method per taxonomic level
        best_by_level = []
        for level in df['level'].unique():
            level_data = df[df['level'] == level]
            if not level_data.empty:
                # Composite score: silhouette (40%) + cev (30%) + speed (20%) + memory (10%)
                level_data = level_data.copy()
                level_data['composite_score'] = (
                    level_data['silhouette_score'] * 0.4 +
                    level_data['final_cev'] * 0.3 +
                    (1 / (1 + level_data['total_time_seconds']/60)) * 0.2 +
                    (1 / (1 + level_data['max_memory_mb']/1000)) * 0.1
                )
                best_idx = level_data['composite_score'].idxmax()
                best_row = level_data.loc[best_idx]
                best_by_level.append({
                    'level': level,
                    'best_method': f"{best_row['method']}-{best_row['device']}",
                    'composite_score': best_row['composite_score'],
                    'silhouette': best_row['silhouette_score'],
                    'cev': best_row['final_cev']
                })

        if best_by_level:
            best_df = pd.DataFrame(best_by_level)
            bars = axes[0,0].bar(best_df['level'], best_df['composite_score'])
            axes[0,0].set_title('Best Method per Taxonomic Level\n(Composite Score)')
            axes[0,0].set_ylabel('Composite Score')
            axes[0,0].tick_params(axis='x', rotation=45)

            # Add method labels on bars
            for bar, method in zip(bars, best_df['best_method']):
                axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              method, ha='center', va='bottom', rotation=90, fontsize=8)

        # 2. Speed vs Quality trade-off
        df_copy = df.copy()
        df_copy['method_device'] = df_copy['method'] + '-' + df_copy['device']

        # Average metrics per method-device
        avg_metrics = df_copy.groupby('method_device').agg({
            'total_time_seconds': 'mean',
            'silhouette_score': 'mean',
            'final_cev': 'mean'
        }).reset_index()

        # Bubble plot: x=time, y=silhouette, size=cev
        scatter = axes[0,1].scatter(avg_metrics['total_time_seconds'],
                                   avg_metrics['silhouette_score'],
                                   s=avg_metrics['final_cev'] * 500,  # Scale bubble size
                                   alpha=0.6, c=range(len(avg_metrics)), cmap='viridis')

        axes[0,1].set_xlabel('Average Time (seconds)')
        axes[0,1].set_ylabel('Average Silhouette Score')
        axes[0,1].set_title('Speed vs Quality Trade-off\n(Bubble size = CEV)')

        # Add method labels
        for i, row in avg_metrics.iterrows():
            axes[0,1].annotate(row['method_device'],
                              (row['total_time_seconds'], row['silhouette_score']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 3. Memory efficiency analysis
        memory_efficiency = df_copy.groupby('method_device').agg({
            'max_memory_mb': 'mean',
            'silhouette_score': 'mean'
        }).reset_index()

        bars3 = axes[1,0].barh(memory_efficiency['method_device'],
                              memory_efficiency['max_memory_mb'])
        axes[1,0].set_title('Memory Usage by Method')
        axes[1,0].set_xlabel('Average Peak Memory (MB)')

        # Color bars by silhouette score
        norm = plt.Normalize(memory_efficiency['silhouette_score'].min(),
                           memory_efficiency['silhouette_score'].max())
        colors = plt.cm.RdYlGn(norm(memory_efficiency['silhouette_score']))

        for bar, color in zip(bars3, colors):
            bar.set_color(color)

        # 4. Overall ranking table (as text)
        axes[1,1].axis('off')

        # Calculate overall ranking
        ranking_data = []
        for method_device in df_copy['method_device'].unique():
            method_data = df_copy[df_copy['method_device'] == method_device]
            if not method_data.empty:
                ranking_data.append({
                    'Method': method_device,
                    'Avg Silhouette': f"{method_data['silhouette_score'].mean():.3f}",
                    'Avg CEV': f"{method_data['final_cev'].mean():.3f}",
                    'Avg Time (s)': f"{method_data['total_time_seconds'].mean():.1f}",
                    'Avg Memory (MB)': f"{method_data['max_memory_mb'].mean():.0f}",
                    'Success Rate': f"{method_data['success'].mean():.1%}"
                })

        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)

            # Create table
            table_text = []
            table_text.append(['Method', 'Silhouette', 'CEV', 'Time', 'Memory', 'Success'])
            for _, row in ranking_df.iterrows():
                table_text.append([
                    row['Method'], row['Avg Silhouette'], row['Avg CEV'],
                    row['Avg Time (s)'], row['Avg Memory (MB)'], row['Success Rate']
                ])

            table = axes[1,1].table(cellText=table_text[1:], colLabels=table_text[0],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)

            axes[1,1].set_title('Performance Summary Table')

        plt.tight_layout()

        filename = f"production_recommendations_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"📊 Production recommendations saved to: {filename}")

    def generate_production_report(self):
        """Generate comprehensive production report"""
        if self.all_results_df is None:
            return

        print(f"\n{'='*80}")
        print(f"🏭 PRODUCTION BENCHMARK REPORT")
        print(f"{'='*80}")

        # Overall statistics
        total_runs = len(self.all_results_df)
        successful_runs = len(self.all_results_df[self.all_results_df['success'] == True])
        success_rate = successful_runs / total_runs if total_runs > 0 else 0

        print(f"📊 Overall Statistics:")
        print(f"   - Total benchmark runs: {total_runs}")
        print(f"   - Successful runs: {successful_runs}")
        print(f"   - Overall success rate: {success_rate:.1%}")

        if successful_runs > 0:
            successful_df = self.all_results_df[self.all_results_df['success'] == True]

            print(f"\n🎯 Performance Summary:")
            print(f"   - Average silhouette score: {successful_df['silhouette_score'].mean():.4f}")
            print(f"   - Average CEV: {successful_df['final_cev'].mean():.3f}")
            print(f"   - Average execution time: {successful_df['total_time_seconds'].mean():.2f}s")
            print(f"   - Average memory usage: {successful_df['max_memory_mb'].mean():.1f}MB")

            # Best performing combinations
            print(f"\n🏆 Top Performing Combinations:")

            # Sort by composite score
            successful_df_copy = successful_df.copy()
            successful_df_copy['composite_score'] = (
                successful_df_copy['silhouette_score'] * 0.4 +
                successful_df_copy['final_cev'] * 0.3 +
                (1 / (1 + successful_df_copy['total_time_seconds']/60)) * 0.2 +
                (1 / (1 + successful_df_copy['max_memory_mb']/1000)) * 0.1
            )

            top_5 = successful_df_copy.nlargest(5, 'composite_score')

            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"   {i}. {row['method'].upper()}-{row['device'].upper()}")
                print(f"      Silhouette: {row['silhouette_score']:.4f}, CEV: {row['final_cev']:.3f}")
                print(f"      Time: {row['total_time_seconds']:.2f}s, Memory: {row['max_memory_mb']:.1f}MB")

        # Save detailed report
        report_filename = f"production_report_detailed_{self.timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(f"Production Benchmark Detailed Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

            # Write summary statistics
            f.write(f"SUMMARY STATISTICS\n")
            f.write(f"Total runs: {total_runs}\n")
            f.write(f"Successful runs: {successful_runs}\n")
            f.write(f"Success rate: {success_rate:.1%}\n\n")

            # Write detailed results
            if not self.all_results_df.empty:
                f.write("DETAILED RESULTS\n")
                f.write(self.all_results_df.to_string(index=False))
                f.write("\n\n")

            # Write feature files information
            if self.production_features:
                f.write("GENERATED FEATURE FILES\n")
                for key, feature_data in self.production_features.items():
                    f.write(f"{key}: {feature_data['features'].shape}\n")

        print(f"\n💾 Detailed report saved to: {report_filename}")

class RKIDataLoader:
    """Specialized data loader untuk struktur file RKI 2025"""

    def __init__(self, 
        base_path: str = "/content/drive/MyDrive/Colab Notebooks/RKI_2025/rki_2025/prep/vectorization"):
        self.base_path = Path(base_path)
        self.data_cache = {}

        # Available taxonomic levels and k-mer sizes from your structure
        self.available_levels = ['class', 'family', 'genus', 'kingdom', 'order', 'phylum', 'species']
        self.available_kmers = [6, 8, 10]

        print(f"🔧 RKI Data Loader initialized")
        print(f"📁 Base path: {self.base_path}")
        print(f"🧬 Available levels: {self.available_levels}")
        print(f"🔢 Available k-mers: {self.available_kmers}")

    def load_single_combination(self, level: str, kmer: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Load single taxonomic level and k-mer combination

        Args:
            level: Taxonomic level (e.g., 'genus', 'species')
            kmer: K-mer size (e.g., 6, 8, 10)

        Returns:
            Tuple of (X_sparse, y_encoded)
        """
        if level not in self.available_levels:
            raise ValueError(f"Level '{level}' not available. Choose from: {self.available_levels}")

        if kmer not in self.available_kmers:
            raise ValueError(f"K-mer '{kmer}' not available. Choose from: {self.available_kmers}")

        print(f"\n🔄 Loading {level.upper()} - K-mer {kmer}")

        # Construct file paths
        level_dir = self.base_path / level / f"k{kmer}"

        X_file = level_dir / f"X_sparse_k{kmer}_{level}.npz"
        y_file = level_dir / f"y_encoded_k{kmer}_{level}.npy"
        label_encoder_file = level_dir / f"label_encoder_k{kmer}_{level}.pkl"
        vectorizer_file = level_dir / f"vectorizer_k{kmer}_{level}.pkl"

        # Check if files exist
        if not X_file.exists():
            raise FileNotFoundError(f"Feature file not found: {X_file}")
        if not y_file.exists():
            raise FileNotFoundError(f"Label file not found: {y_file}")

        try:
            # Load sparse matrix
            print(f"📁 Loading features from: {X_file}")
            X_data = np.load(X_file)

            if 'data' in X_data and 'indices' in X_data and 'indptr' in X_data:
                # Proper sparse matrix format
                X_sparse = sparse.csr_matrix((X_data['data'], X_data['indices'], X_data['indptr']),
                                           shape=X_data['shape'])
            else:
                # Fallback: try to load as dense and convert
                X_sparse = sparse.csr_matrix(X_data['arr_0'])

            print(f"✅ Features loaded: {X_sparse.shape}")
            print(f"   Density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.6f}")

            # Load labels
            print(f"📁 Loading labels from: {y_file}")
            y_encoded = np.load(y_file)

            print(f"✅ Labels loaded: {len(y_encoded)} samples")
            print(f"   Unique classes: {len(np.unique(y_encoded))}")
            print(f"   Class distribution: {np.bincount(y_encoded)[:10]}{'...' if len(np.unique(y_encoded)) > 10 else ''}")

            # Validate dimensions
            if X_sparse.shape[0] != len(y_encoded):
                print(f"⚠️  Dimension mismatch: X={X_sparse.shape[0]}, y={len(y_encoded)}")
                min_size = min(X_sparse.shape[0], len(y_encoded))
                X_sparse = X_sparse[:min_size]
                y_encoded = y_encoded[:min_size]
                print(f"   Adjusted to: {min_size} samples")

            # Load additional metadata if needed
            metadata = {}
            if label_encoder_file.exists():
                print(f"📁 Loading label encoder from: {label_encoder_file}")
                with open(label_encoder_file, 'rb') as f:
                    metadata['label_encoder'] = pickle.load(f)

            if vectorizer_file.exists():
                print(f"📁 Loading vectorizer from: {vectorizer_file}")
                with open(vectorizer_file, 'rb') as f:
                    metadata['vectorizer'] = pickle.load(f)

            # Cache the result
            cache_key = (level, kmer)
            self.data_cache[cache_key] = (X_sparse, y_encoded, metadata)

            print(f"✅ Successfully loaded {level} k-mer {kmer}")
            return X_sparse, y_encoded

        except Exception as e:
            print(f"❌ Error loading {level} k-mer {kmer}: {str(e)}")
            raise e

    def load_multiple_combinations(self, combinations: List[Tuple[str, int]]) -> Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]:
        """
        Load multiple taxonomic level and k-mer combinations

        Args:
            combinations: List of (level, kmer) tuples

        Returns:
            Dictionary with format {(level, kmer): (X_sparse, y)}
        """
        print(f"\n🔄 Loading {len(combinations)} combinations...")

        data_dict = {}
        successful_loads = 0

        for level, kmer in combinations:
            try:
                X_sparse, y_encoded = self.load_single_combination(level, kmer)
                data_dict[(level, kmer)] = (X_sparse, y_encoded)
                successful_loads += 1

            except Exception as e:
                print(f"❌ Failed to load {level} k-mer {kmer}: {str(e)}")
                continue

        print(f"\n✅ Successfully loaded {successful_loads}/{len(combinations)} combinations")
        return data_dict

    def load_all_available_combinations(self) -> Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]:
        """Load all available combinations"""
        all_combinations = [(level, kmer) for level in self.available_levels for kmer in self.available_kmers]
        return self.load_multiple_combinations(all_combinations)

    def load_single_level_all_kmers(self, level: str) -> Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]:
        """Load single taxonomic level with all k-mer sizes"""
        combinations = [(level, kmer) for kmer in self.available_kmers]
        return self.load_multiple_combinations(combinations)

    def load_single_kmer_all_levels(self, kmer: int) -> Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]:
        """Load single k-mer size with all taxonomic levels"""
        combinations = [(level, kmer) for level in self.available_levels]
        return self.load_multiple_combinations(combinations)

    def get_metadata(self, level: str, kmer: int) -> Dict:
        """Get metadata for specific combination"""
        cache_key = (level, kmer)
        if cache_key in self.data_cache:
            return self.data_cache[cache_key][2]
        else:
            # Load if not in cache
            self.load_single_combination(level, kmer)
            return self.data_cache[cache_key][2]

    def get_class_names(self, level: str, kmer: int) -> Optional[List[str]]:
        """Get original class names using label encoder"""
        try:
            metadata = self.get_metadata(level, kmer)
            if 'label_encoder' in metadata:
                return list(metadata['label_encoder'].classes_)
            return None
        except:
            return None
    
    def load_train_data(self, level: str, kmer: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Load training data for specific level and k-mer
        
        Args:
            level: Taxonomic level (e.g., 'genus', 'species')
            kmer: K-mer size (e.g., 6, 8, 10)
            
        Returns:
            Tuple of (X_train_sparse, y_train)
        """
        if level not in self.available_levels:
            raise ValueError(f"Level '{level}' not available. Choose from: {self.available_levels}")

        if kmer not in self.available_kmers:
            raise ValueError(f"K-mer '{kmer}' not available. Choose from: {self.available_kmers}")

        print(f"📂 Loading TRAIN data for {level.upper()} - K-mer {kmer}")

        # Construct file paths for training data
        level_dir = self.base_path / level / f"k{kmer}"

        X_train_file = level_dir / f"X_train_sparse_k{kmer}_{level}.npz"
        y_train_file = level_dir / f"y_train_k{kmer}_{level}.npy"

        # Check if files exist
        if not X_train_file.exists():
            raise FileNotFoundError(f"Training feature file not found: {X_train_file}")
        if not y_train_file.exists():
            raise FileNotFoundError(f"Training label file not found: {y_train_file}")

        try:
            # Load sparse matrix
            print(f"📁 Loading train features from: {X_train_file}")
            X_data = np.load(X_train_file)

            if 'data' in X_data and 'indices' in X_data and 'indptr' in X_data:
                # Proper sparse matrix format
                X_train_sparse = sparse.csr_matrix((X_data['data'], X_data['indices'], X_data['indptr']),
                                                 shape=X_data['shape'])
            else:
                # Fallback: try to load as dense and convert
                X_train_sparse = sparse.csr_matrix(X_data['arr_0'])

            print(f"✅ Train features loaded: {X_train_sparse.shape}")

            # Load labels
            print(f"📁 Loading train labels from: {y_train_file}")
            y_train = np.load(y_train_file)

            print(f"✅ Train labels loaded: {len(y_train)} samples")
            print(f"   Unique classes: {len(np.unique(y_train))}")

            # Validate dimensions
            if X_train_sparse.shape[0] != len(y_train):
                print(f"⚠️  Dimension mismatch: X={X_train_sparse.shape[0]}, y={len(y_train)}")
                min_size = min(X_train_sparse.shape[0], len(y_train))
                X_train_sparse = X_train_sparse[:min_size]
                y_train = y_train[:min_size]
                print(f"   Adjusted to: {min_size} samples")

            print(f"✅ Successfully loaded TRAIN data for {level} k-mer {kmer}")
            return X_train_sparse, y_train

        except Exception as e:
            print(f"❌ Error loading train data for {level} k-mer {kmer}: {str(e)}")
            raise e
    
    def load_test_data(self, level: str, kmer: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Load test data for specific level and k-mer
        
        Args:
            level: Taxonomic level (e.g., 'genus', 'species')
            kmer: K-mer size (e.g., 6, 8, 10)
            
        Returns:
            Tuple of (X_test_sparse, y_test)
        """
        if level not in self.available_levels:
            raise ValueError(f"Level '{level}' not available. Choose from: {self.available_levels}")

        if kmer not in self.available_kmers:
            raise ValueError(f"K-mer '{kmer}' not available. Choose from: {self.available_kmers}")

        print(f"📂 Loading TEST data for {level.upper()} - K-mer {kmer}")

        # Construct file paths for test data
        level_dir = self.base_path / level / f"k{kmer}"

        X_test_file = level_dir / f"X_test_sparse_k{kmer}_{level}.npz"
        y_test_file = level_dir / f"y_test_k{kmer}_{level}.npy"

        # Check if files exist
        if not X_test_file.exists():
            raise FileNotFoundError(f"Test feature file not found: {X_test_file}")
        if not y_test_file.exists():
            raise FileNotFoundError(f"Test label file not found: {y_test_file}")

        try:
            # Load sparse matrix
            print(f"📁 Loading test features from: {X_test_file}")
            X_data = np.load(X_test_file)

            if 'data' in X_data and 'indices' in X_data and 'indptr' in X_data:
                # Proper sparse matrix format
                X_test_sparse = sparse.csr_matrix((X_data['data'], X_data['indices'], X_data['indptr']),
                                                shape=X_data['shape'])
            else:
                # Fallback: try to load as dense and convert
                X_test_sparse = sparse.csr_matrix(X_data['arr_0'])

            print(f"✅ Test features loaded: {X_test_sparse.shape}")

            # Load labels
            print(f"📁 Loading test labels from: {y_test_file}")
            y_test = np.load(y_test_file)

            print(f"✅ Test labels loaded: {len(y_test)} samples")
            print(f"   Unique classes: {len(np.unique(y_test))}")

            # Validate dimensions
            if X_test_sparse.shape[0] != len(y_test):
                print(f"⚠️  Dimension mismatch: X={X_test_sparse.shape[0]}, y={len(y_test)}")
                min_size = min(X_test_sparse.shape[0], len(y_test))
                X_test_sparse = X_test_sparse[:min_size]
                y_test = y_test[:min_size]
                print(f"   Adjusted to: {min_size} samples")

            print(f"✅ Successfully loaded TEST data for {level} k-mer {kmer}")
            return X_test_sparse, y_test

        except Exception as e:
            print(f"❌ Error loading test data for {level} k-mer {kmer}: {str(e)}")
            raise e

    def print_data_summary(self, data_dict: Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]):
        """Print comprehensive data summary"""
        print(f"\n{'='*80}")
        print(f"📊 DATA SUMMARY")
        print(f"{'='*80}")

        total_combinations = len(data_dict)
        total_samples = sum(X.shape[0] for X, y in data_dict.values())
        total_features = sum(X.shape[1] for X, y in data_dict.values())

        print(f"🔢 Total combinations: {total_combinations}")
        print(f"👥 Total samples across all combinations: {total_samples:,}")
        print(f"🧬 Total features across all combinations: {total_features:,}")

        print(f"\n📋 Detailed breakdown:")
        print(f"{'Level':<10} {'K-mer':<6} {'Samples':<10} {'Features':<12} {'Classes':<8} {'Density':<10}")
        print(f"{'-'*70}")

        for (level, kmer), (X, y) in sorted(data_dict.items()):
            density = X.nnz / (X.shape[0] * X.shape[1])
            n_classes = len(np.unique(y))

            print(f"{level:<10} {kmer:<6} {X.shape[0]:<10,} {X.shape[1]:<12,} {n_classes:<8} {density:<10.6f}")

        print(f"{'-'*70}")
    
    def create_data_dict(self, 
                        levels: List[str] = None, 
                        kmers: List[int] = None) -> Dict[Tuple[str, int], Tuple[sparse.csr_matrix, np.ndarray]]:
        """
        Create data dictionary from RKI structure using this loader instance
        
        Args:
            levels: List of taxonomic levels (None for all available)
            kmers: List of k-mer sizes (None for all available)
            
        Returns:
            Data dictionary ready for benchmark
            
        Example:
            loader = RKIDataLoader('/path/to/data')
            
            # Load specific combinations
            data_dict = loader.create_data_dict(
                levels=['genus', 'species'],
                kmers=[6, 8]
            )
            
            # Load all available
            data_dict = loader.create_data_dict()
        """
        # Use all available if not specified
        if levels is None:
            levels = self.available_levels
        if kmers is None:
            kmers = self.available_kmers
            
        print(f"\n🔄 Creating data dictionary...")
        print(f"📋 Target levels: {levels}")
        print(f"🔢 Target k-mers: {kmers}")
        
        # Create combinations
        combinations = [(level, kmer) for level in levels for kmer in kmers]
        
        print(f"🎯 Total combinations to load: {len(combinations)}")
        
        # Load combinations using existing method
        data_dict = self.load_multiple_combinations(combinations)
        
        if data_dict:
            print(f"✅ Successfully created data dictionary with {len(data_dict)} combinations")
            self.print_data_summary(data_dict)
        else:
            print(f"❌ Failed to create data dictionary")
            
        return data_dict

class CustomBenchmarkRunner:
    """Runner untuk benchmark dengan custom path dan semua metode"""

    def __init__(self, custom_base_path: str):
        self.custom_base_path = Path(custom_base_path)
        self.loader = RKIDataLoader(str(self.custom_base_path))
        self.benchmark_results = {}

        print(f"🚀 Custom Benchmark Runner Initialized")
        print(f"📁 Custom base path: {self.custom_base_path}")

        # Verify path exists
        if not self.custom_base_path.exists():
            raise FileNotFoundError(f"Path does not exist: {self.custom_base_path}")

        # Check available data
        self.check_available_data()

    def check_available_data(self):
        """Check what data is available in the custom path"""
        print(f"\n🔍 Checking available data...")

        available_combinations = []

        for level in self.loader.available_levels:
            for kmer in self.loader.available_kmers:
                level_dir = self.custom_base_path / level / f"k{kmer}"
                X_file = level_dir / f"X_sparse_k{kmer}_{level}.npz"
                y_file = level_dir / f"y_encoded_k{kmer}_{level}.npy"

                if X_file.exists() and y_file.exists():
                    available_combinations.append((level, kmer))
                    print(f"   ✅ {level} k-mer {kmer}")
                else:
                    print(f"   ❌ {level} k-mer {kmer} (missing files)")

        self.available_combinations = available_combinations
        print(f"\n📊 Total available combinations: {len(self.available_combinations)}")

        if not self.available_combinations:
            raise ValueError("No valid data combinations found!")

    def run_full_benchmark_all_methods(self,
                                      levels: List[str] = None,
                                      kmers: List[int] = None,
                                      methods: List[str] = None,
                                      devices: List[str] = None,
                                      **benchmark_kwargs) -> ProductionDimensionalityBenchmark:
        """
        Run comprehensive benchmark for all specified methods

        Args:
            levels: List of taxonomic levels (None for all available)
            kmers: List of k-mer sizes (None for all available)
            methods: List of methods (None for all: ['ipca', 'svd', 'umap', 'autoencoder'])
            devices: List of devices (None for all: ['cpu', 'gpu'])
            **benchmark_kwargs: Additional benchmark parameters

        Returns:
            Benchmark instance with results
        """
        print(f"\n🏭 STARTING FULL BENCHMARK WITH ALL METHODS")
        print(f"{'='*80}")

        # Set defaults if not specified
        if levels is None:
            levels = list(set([combo[0] for combo in self.available_combinations]))
        if kmers is None:
            kmers = list(set([combo[1] for combo in self.available_combinations]))
        if methods is None:
            methods = ['ipca', 'svd', 'umap', 'autoencoder']
        if devices is None:
            devices = ['cpu', 'gpu']

        print(f"🧬 Target levels: {levels}")
        print(f"🔢 Target k-mers: {kmers}")
        print(f"⚙️  Target methods: {methods}")
        print(f"💻 Target devices: {devices}")

        # Filter available combinations
        target_combinations = [(l, k) for l, k in self.available_combinations
                              if l in levels and k in kmers]

        print(f"🎯 Target combinations: {len(target_combinations)}")
        for level, kmer in target_combinations:
            print(f"   - {level} k-mer {kmer}")

        if not target_combinations:
            raise ValueError("No valid combinations found for specified criteria!")

        # Load data using improved method
        print(f"\n📁 Loading data for {len(target_combinations)} combinations...")
        data_dict = self.loader.create_data_dict(
            levels=levels,
            kmers=kmers
        )

        if not data_dict:
            raise ValueError("Failed to load any data!")

        # Print data summary
        self.loader.print_data_summary(data_dict)

        # Set benchmark parameters
        default_params = {
            'cev_threshold': 0.95,
            'start_components': 100,
            'step_components': 50,
            'max_limit': 1000,
            'batch_size': 500,
            'cv_folds': 3,
            'autoencoder_epochs': 50,
            'umap_n_neighbors': 15,
            'taxonomic_levels': levels,
            'kmer_sizes': kmers
        }
        default_params.update(benchmark_kwargs)

        print(f"\n⚙️  Benchmark Parameters:")
        for key, value in default_params.items():
            print(f"   {key}: {value}")

        # Create custom benchmark instance
        benchmark = ProductionDimensionalityBenchmark(**default_params)

        # Override available combinations to match our methods/devices
        original_combinations = benchmark.get_available_combinations()
        filtered_combinations = [(method, device) for method, device in original_combinations
                               if method in methods and device in devices]

        print(f"\n🔧 Available method-device combinations:")
        for method, device in filtered_combinations:
            print(f"   - {method.upper()}-{device.upper()}")

        # Monkey patch to use only our specified combinations
        benchmark.get_available_combinations = lambda: filtered_combinations

        # Run benchmark
        print(f"\n🚀 Starting benchmark...")
        start_time = time.time()

        try:
            results = benchmark.run_production_benchmark(data_dict)

            total_time = time.time() - start_time
            print(f"\n✅ Benchmark completed successfully!")
            print(f"⏱️  Total execution time: {total_time/60:.2f} minutes")

            # Store results
            self.benchmark_results[f"full_benchmark_{int(time.time())}"] = benchmark

            return benchmark

        except Exception as e:
            print(f"❌ Benchmark failed: {str(e)}")
            raise e

    def run_quick_benchmark_sample(self,
                                  sample_level: str = 'genus',
                                  sample_kmer: int = 6,
                                  methods: List[str] = None,
                                  **benchmark_kwargs) -> ProductionDimensionalityBenchmark:
        """
        Run quick benchmark on single combination to test setup

        Args:
            sample_level: Sample taxonomic level
            sample_kmer: Sample k-mer size
            methods: Methods to test
            **benchmark_kwargs: Additional parameters

        Returns:
            Benchmark instance
        """
        print(f"\n🔬 QUICK BENCHMARK TEST")
        print(f"{'='*50}")

        if methods is None:
            methods = ['ipca', 'svd']  # Quick methods for testing

        # Check if sample combination is available
        if (sample_level, sample_kmer) not in self.available_combinations:
            print(f"⚠️  {sample_level} k-mer {sample_kmer} not available")
            print(f"📋 Available combinations: {self.available_combinations[:5]}...")
            # Use first available
            sample_level, sample_kmer = self.available_combinations[0]
            print(f"🔄 Using {sample_level} k-mer {sample_kmer} instead")

        # Load single combination using improved method
        data_dict = self.loader.create_data_dict(
            levels=[sample_level],
            kmers=[sample_kmer]
        )

        # Quick benchmark parameters
        quick_params = {
            'cv_folds': 2,
            'start_components': 50,
            'max_limit': 200,
            'autoencoder_epochs': 20,
            'taxonomic_levels': [sample_level],
            'kmer_sizes': [sample_kmer]
        }
        quick_params.update(benchmark_kwargs)

        print(f"🎯 Testing with {sample_level} k-mer {sample_kmer}")
        print(f"⚙️  Methods: {methods}")

        # Create benchmark
        benchmark = ProductionDimensionalityBenchmark(**quick_params)

        # Filter methods
        original_combinations = benchmark.get_available_combinations()
        filtered_combinations = [(method, device) for method, device in original_combinations
                               if method in methods]
        benchmark.get_available_combinations = lambda: filtered_combinations

        # Run quick benchmark
        results = benchmark.run_production_benchmark(data_dict)

        print(f"✅ Quick benchmark completed")
        return benchmark

    def run_method_comparison(self,
                             test_level: str = 'genus',
                             test_kmer: int = 6,
                             **benchmark_kwargs) -> Dict[str, ProductionDimensionalityBenchmark]:
        """
        Run comparison of all methods on single dataset

        Args:
            test_level: Level to test
            test_kmer: K-mer to test
            **benchmark_kwargs: Additional parameters

        Returns:
            Dictionary of benchmark results per method
        """
        print(f"\n⚔️  METHOD COMPARISON")
        print(f"{'='*50}")

        methods = ['ipca', 'svd', 'umap', 'autoencoder']
        comparison_results = {}

        # Load test data using improved method
        data_dict = self.loader.create_data_dict(
            levels=[test_level],
            kmers=[test_kmer]
        )

        print(f"🎯 Comparing methods on {test_level} k-mer {test_kmer}")

        # Test each method individually
        for method in methods:
            print(f"\n🔄 Testing {method.upper()}...")

            try:
                # Method-specific parameters
                method_params = {
                    'cv_folds': 2,
                    'start_components': 50,
                    'max_limit': 200,
                    'taxonomic_levels': [test_level],
                    'kmer_sizes': [test_kmer]
                }

                if method == 'autoencoder':
                    method_params['autoencoder_epochs'] = 30
                elif method == 'umap':
                    method_params['umap_n_neighbors'] = 15

                method_params.update(benchmark_kwargs)

                # Create benchmark for this method
                benchmark = ProductionDimensionalityBenchmark(**method_params)

                # Filter to only this method
                filtered_combinations = [(method, 'cpu')]
                if method in ['ipca', 'svd', 'umap', 'autoencoder']:
                    filtered_combinations.append((method, 'gpu'))

                benchmark.get_available_combinations = lambda: filtered_combinations

                # Run benchmark
                benchmark.run_production_benchmark(data_dict)
                comparison_results[method] = benchmark

                print(f"   ✅ {method.upper()} completed")

            except Exception as e:
                print(f"   ❌ {method.upper()} failed: {str(e)}")
                continue

        # Create comparison report
        self.create_method_comparison_report(comparison_results, test_level, test_kmer)

        return comparison_results

    # Modifikasi create_method_comparison_report untuk include manifold metrics

    def create_method_comparison_report(self,
                                    comparison_results: Dict[str, ProductionDimensionalityBenchmark],
                                    test_level: str,
                                    test_kmer: int):
        """Enhanced comparison report with manifold metrics"""
        print(f"\n📊 ENHANCED METHOD COMPARISON REPORT")
        print(f"{'='*80}")
        print(f"Dataset: {test_level} k-mer {test_kmer}")
        print(f"{'='*80}")

        if not comparison_results:
            print("❌ No successful results to compare")
            return

        # Collect enhanced metrics
        metrics_data = []

        for method, benchmark in comparison_results.items():
            if benchmark.all_results_df is not None and not benchmark.all_results_df.empty:
                successful_results = benchmark.all_results_df[benchmark.all_results_df['success'] == True]

                if not successful_results.empty:
                    # Traditional metrics
                    avg_silhouette = successful_results['silhouette_score'].mean()
                    avg_cev = successful_results['final_cev'].mean()
                    avg_time = successful_results['total_time_seconds'].mean()
                    avg_memory = successful_results['max_memory_mb'].mean()
                    success_rate = len(successful_results) / len(benchmark.all_results_df)

                    # Enhanced manifold metrics (if available)
                    manifold_metrics = {}
                    if 'trustworthiness_score' in successful_results.columns:
                        manifold_metrics['avg_trustworthiness'] = successful_results['trustworthiness_score'].mean()
                    if 'continuity_score' in successful_results.columns:
                        manifold_metrics['avg_continuity'] = successful_results['continuity_score'].mean()
                    if 'manifold_combined_score' in successful_results.columns:
                        manifold_metrics['avg_manifold_score'] = successful_results['manifold_combined_score'].mean()
                    if 'optimization_method' in successful_results.columns:
                        manifold_metrics['optimization_method'] = successful_results['optimization_method'].iloc[0]

                    # Base metrics
                    method_data = {
                        'Method': method.upper(),
                        'Avg Silhouette': f"{avg_silhouette:.4f}",
                        'Avg CEV': f"{avg_cev:.3f}",
                        'Avg Time (s)': f"{avg_time:.2f}",
                        'Avg Memory (MB)': f"{avg_memory:.1f}",
                        'Success Rate': f"{success_rate:.1%}"
                    }

                    # Add manifold metrics if available
                    if manifold_metrics:
                        if 'avg_trustworthiness' in manifold_metrics:
                            method_data['Trustworthiness'] = f"{manifold_metrics['avg_trustworthiness']:.4f}"
                        if 'avg_continuity' in manifold_metrics:
                            method_data['Continuity'] = f"{manifold_metrics['avg_continuity']:.4f}"
                        if 'avg_manifold_score' in manifold_metrics:
                            method_data['Manifold Score'] = f"{manifold_metrics['avg_manifold_score']:.4f}"
                        if 'optimization_method' in manifold_metrics:
                            method_data['Optimization'] = manifold_metrics['optimization_method']

                    metrics_data.append(method_data)

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            print(df.to_string(index=False))

            # Save enhanced report
            report_filename = f"enhanced_method_comparison_{test_level}_k{test_kmer}_{int(time.time())}.csv"
            df.to_csv(report_filename, index=False)
            print(f"\n💾 Enhanced comparison report saved: {report_filename}")

            # Print method-specific insights
            print(f"\n🧠 METHOD-SPECIFIC INSIGHTS:")
            for method_data in metrics_data:
                method = method_data['Method']
                print(f"\n   {method}:")

                if 'Optimization' in method_data:
                    opt_method = method_data['Optimization']
                    print(f"     - Optimization method: {opt_method}")

                    if opt_method == 'manifold':
                        print(f"     - Trustworthiness: {method_data.get('Trustworthiness', 'N/A')}")
                        print(f"     - Continuity: {method_data.get('Continuity', 'N/A')}")
                        print(f"     - Manifold Score: {method_data.get('Manifold Score', 'N/A')}")
                    elif opt_method == 'cev':
                        print(f"     - CEV-based optimization")
                        print(f"     - Final CEV: {method_data['Avg CEV']}")

                print(f"     - Silhouette Score: {method_data['Avg Silhouette']}")
                print(f"     - Execution Time: {method_data['Avg Time (s)']}s")
                print(f"     - Memory Usage: {method_data['Avg Memory (MB)']}MB")
        else:
            print("❌ No successful results to report")

def benchmark_(custom_path = "src/data",  # Fixed: Changed to local workspace path
            levels = ['genus', 'species'],
            kmers = [6],
            methods = ['ipca', 'svd', 'umap', 'autoencoder'],
            devices = ['cpu', 'gpu'],
            enable_manifold_metrics = True,
            cv_folds = 2,
            autoencoder_epochs = 30,
            cev_threshold = 0.95,
            umap_n_neighbors = 15,
            output_directory = None,
            skip_existing = True,
            **kwargs  # Added: Accept additional parameters
            ):
    """Example using enhanced manifold learning metrics with organized output
    Args:
        custom_path: Path to vectorization directory
        levels: Taxonomic levels to include
        kmers: K-mer sizes to include
        methods: Methods to test
        devices: Devices to test
        enable_manifold_metrics: Enable manifold learning metrics
        cv_folds: Number of cross-validation folds
        autoencoder_epochs: Number of epochs for autoencoder training
        cev_threshold: CEV threshold for optimization
        umap_n_neighbors: UMAP n_neighbors parameter
        output_directory: Base directory for organized outputs (default: /Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction)
        skip_existing: If True, skip existing files; if False, overwrite
    Returns:
        Benchmark results with enhanced metrics and organized file outputs
    Example:
        benchmark = benchmark_(
            custom_path='/your/custom/path',
            levels=['genus', 'species'],
            kmers=[6, 8],
            methods=['ipca', 'svd', 'umap', 'autoencoder'],
            devices=['cpu', 'gpu'],
            enable_manifold_metrics=True,
            cv_folds=2,
            autoencoder_epochs=30,
            cev_threshold=0.95,
            umap_n_neighbors=15,
            output_directory='/custom/output/path',
            skip_existing=True
        )
    """

    print("🚀 Example: Enhanced benchmark with manifold metrics")

    try:
        # Fixed: Use correct variable name and method
        runner = CustomBenchmarkRunner(custom_path)  # Fixed: Use custom_path instead of custom_base_path
        benchmark = runner.run_full_benchmark_all_methods(  # Fixed: Use existing method
            levels = levels,
            kmers = kmers,
            methods = methods,  # All methods
            devices = devices,
            enable_manifold_metrics = enable_manifold_metrics,
            cv_folds = cv_folds,
            autoencoder_epochs = autoencoder_epochs,
            cev_threshold = cev_threshold,
            umap_n_neighbors = umap_n_neighbors,
            output_directory = output_directory,
            skip_existing = skip_existing,
            **kwargs  # Pass additional parameters
        )

        print("✅ Enhanced benchmark completed!")

        # Print enhanced results summary
        if benchmark.all_results_df is not None:
            print(f"\n📊 ENHANCED RESULTS SUMMARY:")

            # Check for manifold metrics
            manifold_cols = ['trustworthiness_score', 'continuity_score', 'manifold_combined_score']
            available_manifold = [col for col in manifold_cols if col in benchmark.all_results_df.columns]

            if available_manifold:
                print(f"🧮 Manifold metrics available: {available_manifold}")

                # Show manifold method results
                manifold_data = benchmark.all_results_df[
                    benchmark.all_results_df['method'].isin(['umap', 'autoencoder']) &
                    (benchmark.all_results_df['success'] == True)
                ]

                if not manifold_data.empty:
                    print(f"\n🌐 MANIFOLD LEARNING RESULTS:")
                    for _, row in manifold_data.iterrows():
                        print(f"   {row['method'].upper()}-{row['device'].upper()}:")
                        print(f"     - Trustworthiness: {row.get('trustworthiness_score', 0):.4f}")
                        print(f"     - Continuity: {row.get('continuity_score', 0):.4f}")
                        print(f"     - Combined Score: {row.get('manifold_combined_score', 0):.4f}")
                        print(f"     - Silhouette: {row['silhouette_score']:.4f}")
            else:
                print("⚠️ No manifold metrics found - check implementation")

        # Generate and save output summary report
        if hasattr(benchmark, 'output_manager'):
            print(f"\n📋 GENERATING OUTPUT SUMMARY REPORT...")
            summary_report = benchmark.output_manager.get_summary_report()
            print(summary_report)
            
            # Save summary report to file
            summary_file = benchmark.output_manager.save_summary_report()
            print(f"💾 Complete output summary saved to: {summary_file}")
        
        return benchmark

    except Exception as e:
        print(f"❌ Enhanced benchmark failed: {str(e)}")
        return None


def benchmark_with_train_test_split(
    custom_path: str,
    levels: List[str] = ['genus', 'species'],
    kmers: List[int] = [6, 8],
    methods: List[str] = ['ipca', 'svd', 'umap', 'autoencoder'],
    devices: List[str] = ['cpu'],
    cv_folds: int = 2,
    output_directory: str = None,
    skip_existing: bool = True,
    **kwargs
):
    """
    Enhanced benchmark function with proper train/test split and organized output
    
    Parameters:
    custom_path: Path to directory containing X_train_sparse and X_test_sparse files
    levels: List of taxonomic levels to process
    kmers: List of k-mer sizes to process  
    methods: List of methods to benchmark
    devices: List of devices to use
    cv_folds: Number of cross-validation folds
    output_directory: Base directory for organized outputs. If None, uses default path
    skip_existing: If True, skip existing files; if False, overwrite
    **kwargs: Additional parameters for ProductionDimensionalityBenchmark
    
    Returns:
    ProductionDimensionalityBenchmark: Configured benchmark instance with results
    
    Output Structure:
    {output_directory}/{level}/{kmer}/{method}/{device}/
    - features_fold{fold}_{components}comp_{timestamp}.csv
    - cev_analysis_fold{fold}_{components}components_{timestamp}.csv  
    - cev_plot_fold{fold}_{components}comp_{timestamp}.png
    - manifold_optimization_fold{fold}_{timestamp}.csv (if applicable)
    """
    
    print(f"🚀 STARTING ENHANCED BENCHMARK WITH TRAIN/TEST SPLIT & ORGANIZED OUTPUT")
    print(f"📂 Data path: {custom_path}")
    print(f"🧬 Levels: {levels}")
    print(f"🧮 K-mers: {kmers}")
    print(f"🔬 Methods: {methods}")
    print(f"🖥️  Devices: {devices}")
    print(f"🔄 CV folds: {cv_folds}")
    
    # Set default output directory if not provided
    if output_directory is None:
        output_directory = '/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction'
    
    print(f"📁 Output directory: {output_directory}")
    print(f"⏭️  Skip existing: {skip_existing}")
    
    # Initialize data loader
    loader = RKIDataLoader(custom_path)
    
    # Load training and test data
    print(f"\n📂 Loading training and test data...")
    train_data_dict = {}
    test_data_dict = {}
    
    for level in levels:
        for kmer in kmers:
            print(f"📊 Loading {level} - K-mer {kmer}...")
            
            try:
                # Load training data
                X_train_sparse, y_train = loader.load_train_data(level, kmer)
                train_data_dict[(level, kmer)] = (X_train_sparse, y_train)
                print(f"   ✅ Train: {X_train_sparse.shape}, Labels: {len(np.unique(y_train))}")
                
                # Load test data
                X_test_sparse, y_test = loader.load_test_data(level, kmer)
                test_data_dict[(level, kmer)] = (X_test_sparse, y_test)
                print(f"   ✅ Test: {X_test_sparse.shape}, Labels: {len(np.unique(y_test))}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {level} - K-mer {kmer}: {str(e)}")
                continue
    
    if not train_data_dict:
        raise ValueError("No data could be loaded! Check your path and file structure.")
    
    print(f"\n✅ Successfully loaded {len(train_data_dict)} combinations")
    
    # Initialize benchmark with output directory settings
    benchmark = ProductionDimensionalityBenchmark(
        cv_folds=cv_folds,
        taxonomic_levels=levels,
        kmer_sizes=kmers,
        output_directory=output_directory,
        skip_existing=skip_existing,
        **kwargs
    )
    
    # Store original method before overriding to avoid recursion
    original_get_combinations = benchmark.get_available_combinations
    
    # Override get_available_combinations to use only specified methods and devices
    def custom_get_available_combinations():
        """Return only the specified methods and devices combinations"""
        available_combinations = []
        
        # Get original combinations using the stored reference
        original_combinations = original_get_combinations()
        
        print(f"🔧 Filtering combinations...")
        print(f"   📋 Requested methods: {methods}")
        print(f"   💻 Requested devices: {devices}")
        print(f"   🔍 Available original combinations: {original_combinations}")
        
        for method in methods:
            for device in devices:
                # Check if this combination is available in original
                if (method, device) in original_combinations:
                    available_combinations.append((method, device))
                    print(f"   ✅ {method.upper()}-{device.upper()} available")
                else:
                    print(f"   ❌ {method.upper()}-{device.upper()} not available")
        
        if not available_combinations:
            raise ValueError(f"No valid combinations found! Requested: {[(m, d) for m in methods for d in devices]}")
        
        print(f"🎯 Final combinations to run: {available_combinations}")
        return available_combinations
    
    # Replace the method with our custom one
    benchmark.get_available_combinations = custom_get_available_combinations
    
    # Run train/test benchmark
    all_results = benchmark.run_production_benchmark_with_test(
        train_data_dict, test_data_dict
    )
    
    # Generate and save output summary report
    if hasattr(benchmark, 'output_manager'):
        print(f"\n📋 GENERATING OUTPUT SUMMARY REPORT...")
        summary_report = benchmark.output_manager.get_summary_report()
        print(summary_report)
        
        # Save summary report to file
        summary_file = benchmark.output_manager.save_summary_report()
        print(f"💾 Complete output summary saved to: {summary_file}")
    
    return benchmark

# =================================================================
# USAGE EXAMPLES
# =================================================================

def example_organized_output_benchmark():
    """
    Contoh penggunaan benchmark dengan output terorganisir dan opsi skip/overwrite
    
    Output Structure akan seperti:
    /Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction/
    ├── genus/
    │   ├── k6/
    │   │   ├── ipca/
    │   │   │   ├── cpu/
    │   │   │   │   ├── features_fold1_100comp_1625097600.csv
    │   │   │   │   ├── cev_analysis_fold1_100components_1625097600.csv
    │   │   │   │   └── cev_plot_fold1_100comp_1625097600.png
    │   │   │   └── gpu/...
    │   │   ├── svd/...
    │   │   ├── umap/...
    │   │   └── autoencoder/...
    │   └── k8/...
    └── species/...
    
    Plus file log:
    - output_paths_log.txt: Log semua file yang disimpan
    - summary_report_{timestamp}.txt: Ringkasan lengkap benchmark
    """
    print("📝 CONTOH PENGGUNAAN BENCHMARK DENGAN OUTPUT TERORGANISIR")
    print("="*80)
    
    # Contoh 1: Benchmark dengan skip existing files (default)
    print("\n🔄 Contoh 1: Skip existing files")
    benchmark1 = benchmark_with_train_test_split(
        custom_path="/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization",
        levels=['genus', 'species'],
        kmers=[6, 8],
        methods=['ipca', 'svd'],
        devices=['cpu'],
        cv_folds=2,
        output_directory='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction',
        skip_existing=True  # Skip jika file sudah ada
    )
    
    # Contoh 2: Benchmark dengan overwrite files
    print("\n🔄 Contoh 2: Overwrite existing files")
    benchmark2 = benchmark_with_train_test_split(
        custom_path="/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization",
        levels=['genus'],
        kmers=[6],
        methods=['ipca'],
        devices=['cpu'],
        cv_folds=1,
        output_directory='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction',
        skip_existing=False  # Timpa file yang sudah ada
    )
    
    # Contoh 3: Menggunakan benchmark_ dengan output terorganisir
    print("\n🔄 Contoh 3: Benchmark dengan manifold metrics")
    benchmark3 = benchmark_(
        custom_path="/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/vectorization",
        levels=['genus'],
        kmers=[6],
        methods=['umap', 'autoencoder'],
        devices=['cpu'],
        enable_manifold_metrics=True,
        cv_folds=2,
        output_directory='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction',
        skip_existing=True
    )
    
    print("\n✅ Semua contoh selesai!")
    print("\n📁 Cek direktori output di: /Users/tirtasetiawan/Documents/rki_v1/rki_2025/prep/reduction")
    print("📋 Cek file log: output_paths_log.txt")
    print("📊 Cek summary report: summary_report_*.txt")

if __name__ == "__main__":
    print("🚀 Feature Reduction Benchmark dengan Output Terorganisir")
    print("Untuk menjalankan contoh, gunakan:")
    print("python feature_reduction.py")
    print("\nAtau import dan gunakan fungsi:")
    print("from benchmark.feature_reduction import benchmark_with_train_test_split")
    print("\n# Jalankan benchmark dengan output terorganisir")
    print("benchmark = benchmark_with_train_test_split(")
    print("    custom_path='path/to/your/data',")
    print("    levels=['genus', 'species'],")
    print("    kmers=[6, 8],")
    print("    methods=['ipca', 'svd', 'umap', 'autoencoder'],")
    print("    devices=['cpu'],")
    print("    output_directory='/path/to/output',")
    print("    skip_existing=True")
    print(")")
    
    # Uncomment line below to run example
    # example_organized_output_benchmark()