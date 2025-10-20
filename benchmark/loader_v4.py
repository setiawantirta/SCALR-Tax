import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_(fasta,
          tax,
          sequence_column_name='sequence',
          taxonomy_column_name='Taxon',
          feature_id_column_name='Feature ID',
          only_bacteria=False):

    import importlib
    import subprocess
    import sys

    # Ensure required packages
    ensure_package('biopython', 'Bio')
    ensure_package('pandas')

    from Bio import SeqIO
    import pandas as pd

    try:
        # Load sequences
        print("Loading FASTA sequences...")
        seqs = {record.id: str(record.seq) for record in SeqIO.parse(fasta, 'fasta')}
        print(f"Loaded {len(seqs)} sequences from FASTA")

        # Load taxonomy
        print("Loading taxonomy data...")
        df_tax = pd.read_csv(tax, sep='\t')
        print(f"Loaded taxonomy data: {len(df_tax)} rows")

        # Check if required columns exist
        if feature_id_column_name not in df_tax.columns:
            raise KeyError(f"Column '{feature_id_column_name}' not found in taxonomy file")

        if taxonomy_column_name not in df_tax.columns:
            raise KeyError(f"Column '{taxonomy_column_name}' not found in taxonomy file")

        # Map sequences to taxonomy
        df_tax[sequence_column_name] = df_tax[feature_id_column_name].map(seqs)
        df_tax = df_tax.dropna(subset=[sequence_column_name])
        print(f"After mapping sequences: {len(df_tax)} rows")

        # Extract taxonomic levels
        print("Extracting taxonomic levels...")

        # Check if taxonomy column has the expected format
        sample_taxon = df_tax[taxonomy_column_name].iloc[0] if len(df_tax) > 0 else ""
        print(f"Sample taxonomy string: {sample_taxon}")

        df_tax['kingdom'] = df_tax[taxonomy_column_name].str.split(';').str[0].fillna('Unclassified')
        df_tax['phylum'] = df_tax[taxonomy_column_name].str.split(';').str[1].fillna('Unclassified')
        df_tax['class'] = df_tax[taxonomy_column_name].str.split(';').str[2].fillna('Unclassified')
        df_tax['order'] = df_tax[taxonomy_column_name].str.split(';').str[3].fillna('Unclassified')
        df_tax['family'] = df_tax[taxonomy_column_name].str.split(';').str[4].fillna('Unclassified')
        df_tax['genus'] = df_tax[taxonomy_column_name].str.split(';').str[5].fillna('Unclassified')
        df_tax['species'] = df_tax[taxonomy_column_name].str.split(';').str[6].fillna('Unclassified')

        # Show unique kingdoms found
        unique_kingdoms = df_tax['kingdom'].unique()
        print(f"Unique kingdoms found: {unique_kingdoms}")

        # Filter by kingdom if needed
        if only_bacteria:
            initial_count = len(df_tax)
            df_tax = df_tax[df_tax['kingdom'].isin(['d__Bacteria'])]
            final_count = len(df_tax)
            print(f"Filtered to bacteria only: {final_count} sequences (removed {initial_count - final_count})")

        print(f"Final dataset: {len(df_tax)} sequences with taxonomy")

        return df_tax

    except Exception as e:
        print(f"Error in load_ function: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise


# Auto install package if missing
def ensure_package(package_name, import_name=None):
    import importlib
    import subprocess
    import sys

    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Installed '{package_name}' successfully.")
        
def plot_freq(df, folder_path=None, level='genus'):
    """Plot frequency distribution for taxonomic level"""
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)

    # Filter out unclassified entries
    df_level_filtered = df[df[level] != 'Unclassified']

    # Calculate frequency counts
    level_counts = df_level_filtered[level].value_counts()

    # Get top and bottom 20
    top_20_level = level_counts.head(20)
    bottom_20_level = level_counts[level_counts > 0].tail(20)

    # Set seaborn style
    sns.set_style("ticks")

    # Plot top 20
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_20_level.values, y=top_20_level.index, palette="viridis")
    plt.title(f"Top 20 most frequent {level}", fontweight='bold')
    plt.xlabel("Frequency", fontweight='bold')
    plt.ylabel(f'{level}', fontweight='bold')
    plt.tight_layout()

    if folder_path:
        file_path = os.path.join(folder_path, f"Top_20_most_frequent_{level}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Plot bottom 20
    plt.figure(figsize=(10, 5))
    sns.barplot(x=bottom_20_level.values, y=bottom_20_level.index, palette="viridis")
    plt.title(f"Bottom 20 least frequent {level}", fontweight='bold')
    plt.xlabel("Frequency", fontweight='bold')
    plt.ylabel(f'{level}', fontweight='bold')
    plt.tight_layout()

    if folder_path:
        file_path = os.path.join(folder_path, f"Bottom_20_least_frequent_{level}.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()

    return file_path

def level_extract_plot_freq(df_tax, path_=None, level='genus', filter_uncultured=True, min_sample_freq=4):
    """Process and plot frequency for specific taxonomic level"""

    print(f"= STEP 1: Load Data {level} =")
    kolom_used = ['sequence', level]

    # Initialize df_filter
    df_filter = None

    if filter_uncultured:
        if level == 'genus':
            delete_pattern = 'g__uncultured'
            df_filter = df_tax[
                (df_tax['genus'] != 'Unclassified') &
                (~df_tax['genus'].str.contains(delete_pattern, case=False, na=False))
            ]

        elif level == 'species':
            delete_patterns = ['s__uncultured', 's__metagenome']
            df_filter = df_tax[
                (df_tax['species'] != 'Unclassified') &
                (~df_tax['species'].str.contains('|'.join(delete_patterns), case=False, na=False))
            ]

        else:
            df_filter = df_tax[df_tax[level] != 'Unclassified']
    else:
        df_filter = df_tax.copy()

    df_filter = df_filter[kolom_used].copy()

    # Summary statistics
    print(f'SUMMARY {level}')
    df_rename = df_filter.rename(columns={level: "label"})
    df_rename.dropna(inplace=True)

    print(f"== Hitung frekuensi kelas & drop label < {min_sample_freq}")
    class_counts = df_rename['label'].value_counts()
    valid_labels = class_counts[class_counts >= min_sample_freq].index
    df_final = df_rename[df_rename['label'].isin(valid_labels)].copy()
    df_final = df_final.reset_index(drop=True)

    # Create folder and save CSV
    csv_level_file_path = None
    folder_path = None

    if path_:
        folder_path = os.path.join(path_, level)
        os.makedirs(folder_path, exist_ok=True)

        csv_level_file_path = os.path.join(folder_path, f'{level}.csv')
        df_final.to_csv(csv_level_file_path, index=False)
        print(f"CSV saved to: {csv_level_file_path}")

    plot_freq(df_final, folder_path=folder_path, level='label')

    print(f"=== Jumlah label sebelum filter: {df_rename['label'].nunique()}")
    print(f"=== Jumlah label setelah filter: {df_final['label'].nunique()}")
    print(f"=== Jumlah baris data: {len(df_final)}")
    print('========================================================')

    return df_final, csv_level_file_path

def load_all_csv_data(paths_list):
    """Load all CSV data from paths list"""
    data_dict = {}

    for path in paths_list:
        # Extract level name from filename
        filename = os.path.basename(path)
        level = filename.replace('.csv', '')

        # Load data
        data_dict[level] = pd.read_csv(path)
        print(f"Loaded {level} data: {len(data_dict[level])} rows")

    return data_dict


# ============================================================
# 🆕 ENHANCED: CENTROID-BASED SAMPLING WITH SMALL CLASS HANDLING
# ============================================================
def centroid_based_sampling(
    df, 
    column_name='sequence', 
    label_column='label', 
    sample_fraction=0.1, 
    min_samples=10, 
    kmer_size=6, 
    method='closest', 
    random_state=42,
    small_class_threshold=10,  # 🆕 NEW
    small_class_strategy='group'  # 🆕 NEW: 'group', 'skip', or 'keep'
):
    """
    Sample data berdasarkan proximity ke centroid tiap class
    WITH SMALL CLASS HANDLING
    
    Parameters:
    -----------
    df : DataFrame
        Data dengan kolom sequence dan label
    column_name : str
        Nama kolom yang berisi sequence
    label_column : str
        Nama kolom yang berisi label
    sample_fraction : float
        Fraction of data to sample per class (0.1 = 10%)
    min_samples : int
        Minimum samples per class setelah sampling
    kmer_size : int
        K-mer size untuk feature extraction (default: 6)
    method : str
        - 'closest': Ambil samples terdekat ke centroid (RECOMMENDED)
        - 'diverse': Ambil samples yang diverse (60% closest + 40% farthest)
        - 'kmeans': Gunakan k-means clustering dalam class
    random_state : int
        Random seed untuk reproducibility
    small_class_threshold : int
        Threshold untuk class kecil (default: 10)
    small_class_strategy : str
        Strategi untuk handle small classes:
        - 'group': Group small classes into 'RARE_CLASS' (RECOMMENDED) ✅
        - 'skip': Skip small classes ⏭️
        - 'keep': Keep all samples from small classes 📦
    
    Returns:
    --------
    df_sampled : DataFrame
        Sampled data yang representatif
    
    Example:
    --------
    >>> df_sampled = centroid_based_sampling(
    ...     df, 
    ...     sample_fraction=0.1, 
    ...     kmer_size=6, 
    ...     method='closest',
    ...     small_class_strategy='group'
    ... )
    """
    from itertools import product
    from sklearn.metrics.pairwise import euclidean_distances
    
    print(f"\n🧬 CENTROID-BASED SAMPLING (ENHANCED)")
    print(f"   Method: {method}")
    print(f"   K-mer size: {kmer_size}")
    print(f"   Sample fraction: {sample_fraction}")
    print(f"   Min samples per class: {min_samples}")
    print(f"   Small class threshold: {small_class_threshold}")
    print(f"   Small class strategy: {small_class_strategy}")
    
    np.random.seed(random_state)
    
    # ============================================================
    # STEP 1: Identify Small Classes
    # ============================================================
    class_counts = df[label_column].value_counts()
    small_classes = class_counts[class_counts <= small_class_threshold].index.tolist()
    large_classes = class_counts[class_counts > small_class_threshold].index.tolist()
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    print(f"   Total classes: {len(class_counts)}")
    print(f"   Large classes (>{small_class_threshold} samples): {len(large_classes)}")
    print(f"   Small classes (≤{small_class_threshold} samples): {len(small_classes)}")
    
    if len(small_classes) > 0:
        total_small_samples = class_counts[small_classes].sum()
        print(f"   Total samples in small classes: {total_small_samples}")
    
    # ============================================================
    # STEP 2: Handle Small Classes
    # ============================================================
    df_large = df[df[label_column].isin(large_classes)].copy()
    df_small = df[df[label_column].isin(small_classes)].copy()
    
    sampled_dfs = []
    
    if len(small_classes) > 0:
        if small_class_strategy == 'group':
            print(f"\n🔷 Grouping {len(small_classes)} small classes into 'RARE_CLASS'")
            df_small_grouped = df_small.copy()
            df_small_grouped[label_column] = 'RARE_CLASS'
            
            # Sample dari grouped rare class
            n_rare_samples = len(df_small_grouped)
            n_target_rare = max(min_samples, int(n_rare_samples * sample_fraction))
            n_target_rare = min(n_target_rare, n_rare_samples)
            
            if n_target_rare < n_rare_samples:
                df_small_sampled = df_small_grouped.sample(n=n_target_rare, random_state=random_state)
                print(f"   ✅ Sampled {n_target_rare} from {n_rare_samples} rare samples")
            else:
                df_small_sampled = df_small_grouped
                print(f"   ✅ Keeping all {n_rare_samples} rare samples")
            
            sampled_dfs.append(df_small_sampled)
        
        elif small_class_strategy == 'skip':
            print(f"\n⏭️  Skipping {len(small_classes)} small classes ({len(df_small)} samples)")
        
        elif small_class_strategy == 'keep':
            print(f"\n📦 Keeping all samples from {len(small_classes)} small classes")
            sampled_dfs.append(df_small)
        
        else:
            raise ValueError(
                f"Unknown small_class_strategy: '{small_class_strategy}'. "
                f"Valid options: 'group', 'skip', 'keep'"
            )
    
    # ============================================================
    # STEP 3: Process Large Classes with Centroid-Based Sampling
    # ============================================================
    if len(large_classes) == 0:
        print("\n⚠️  WARNING: No large classes found!")
        if len(sampled_dfs) > 0:
            df_sampled = pd.concat(sampled_dfs, ignore_index=True)
            return df_sampled
        else:
            print("❌ ERROR: No data to return!")
            return df.head(0)  # Return empty dataframe with same structure
    
    print(f"\n🔬 Processing {len(large_classes)} large classes...")
    
    # Generate k-mer vocabulary
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(bases, repeat=kmer_size)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    n_features = len(all_kmers)
    
    print(f"   📊 K-mer feature space: {n_features} dimensions")
    
    # Process each large class
    for label_idx, label in enumerate(large_classes, 1):
        df_class = df_large[df_large[label_column] == label].copy()
        n_samples_class = len(df_class)
        
        # Calculate target samples
        n_target = max(min_samples, int(n_samples_class * sample_fraction))
        n_target = min(n_target, n_samples_class)
        
        if label_idx <= 3 or label_idx == len(large_classes):
            print(f"\n   [{label_idx}/{len(large_classes)}] Class: {label[:50]}")
            print(f"      Original: {n_samples_class} → Target: {n_target} samples")
        elif label_idx == 4:
            print(f"\n   ... processing remaining classes ...")
        
        # ============================================================
        # Safety Check
        # ============================================================
        if n_samples_class == 0:
            print(f"      ⚠️  WARNING: Class has 0 samples, skipping...")
            continue
        
        if n_target == 0:
            print(f"      ⚠️  WARNING: Target is 0, using minimum 1 sample")
            n_target = 1
        
        # ============================================================
        # Extract K-mer Features
        # ============================================================
        try:
            sequences = df_class[column_name].values
            X_class = np.zeros((n_samples_class, n_features), dtype=np.float32)
            
            for seq_idx, seq in enumerate(sequences):
                kmer_counts = Counter()
                seq_upper = str(seq).upper()
                
                # Count k-mers
                for i in range(len(seq_upper) - kmer_size + 1):
                    kmer = seq_upper[i:i+kmer_size]
                    if kmer in kmer_to_idx:
                        kmer_counts[kmer] += 1
                
                # Convert to feature vector
                for kmer, count in kmer_counts.items():
                    X_class[seq_idx, kmer_to_idx[kmer]] = count
                
                # Normalize (L1 normalization)
                row_sum = X_class[seq_idx].sum()
                if row_sum > 0:
                    X_class[seq_idx] /= row_sum
            
            # ============================================================
            # Check if feature matrix is valid
            # ============================================================
            if X_class.shape[0] == 0:
                print(f"      ⚠️  WARNING: Feature matrix has 0 rows, skipping...")
                continue
            
            # ============================================================
            # Calculate Centroid
            # ============================================================
            centroid = X_class.mean(axis=0, keepdims=True)
            
            # ============================================================
            # Sample Based on Method
            # ============================================================
            
            if method == 'closest':
                distances = euclidean_distances(X_class, centroid).flatten()
                closest_indices = np.argsort(distances)[:n_target]
                df_sampled_class = df_class.iloc[closest_indices].copy()
                
                if label_idx <= 3 or label_idx == len(large_classes):
                    avg_dist = distances[closest_indices].mean()
                    print(f"      ✅ Avg distance to centroid: {avg_dist:.4f}")
            
            elif method == 'diverse':
                distances = euclidean_distances(X_class, centroid).flatten()
                
                n_close = max(1, int(n_target * 0.6))
                n_far = max(1, n_target - n_close)
                
                sorted_indices = np.argsort(distances)
                closest_indices = sorted_indices[:n_close]
                farthest_indices = sorted_indices[-n_far:]
                
                selected_indices = np.concatenate([closest_indices, farthest_indices])
                df_sampled_class = df_class.iloc[selected_indices].copy()
                
                if label_idx <= 3 or label_idx == len(large_classes):
                    print(f"      ✅ Close: {n_close}, Far: {n_far}")
            
            elif method == 'kmeans':
                from sklearn.cluster import KMeans
                
                n_clusters = min(5, n_target, max(2, n_samples_class // 10))
                
                if n_clusters <= 1 or n_samples_class < 2:
                    # Fallback to closest
                    distances = euclidean_distances(X_class, centroid).flatten()
                    closest_indices = np.argsort(distances)[:n_target]
                    df_sampled_class = df_class.iloc[closest_indices].copy()
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(X_class)
                    
                    selected_indices = []
                    samples_per_cluster = max(1, n_target // n_clusters)
                    remainder = n_target % n_clusters
                    
                    for cluster_id in range(n_clusters):
                        cluster_indices = np.where(cluster_labels == cluster_id)[0]
                        
                        if len(cluster_indices) == 0:
                            continue
                        
                        n_sample_cluster = samples_per_cluster + (1 if cluster_id < remainder else 0)
                        n_sample_cluster = min(n_sample_cluster, len(cluster_indices))
                        
                        # Get closest to cluster center
                        cluster_center = kmeans.cluster_centers_[cluster_id:cluster_id+1]
                        cluster_distances = euclidean_distances(
                            X_class[cluster_indices], 
                            cluster_center
                        ).flatten()
                        
                        closest_in_cluster = np.argsort(cluster_distances)[:n_sample_cluster]
                        selected_indices.extend(cluster_indices[closest_in_cluster])
                    
                    df_sampled_class = df_class.iloc[selected_indices].copy()
                    
                    if label_idx <= 3 or label_idx == len(large_classes):
                        print(f"      ✅ Sampled from {n_clusters} clusters")
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            sampled_dfs.append(df_sampled_class)
            
        except Exception as e:
            print(f"      ❌ ERROR processing class: {e}")
            print(f"         Skipping this class...")
            continue
    
    # ============================================================
    # STEP 4: Combine All Sampled Data
    # ============================================================
    if len(sampled_dfs) == 0:
        print("\n❌ ERROR: No classes were successfully sampled!")
        return df.head(0)
    
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n📈 SAMPLING SUMMARY:")
    print(f"   Original: {len(df):,} samples, {df[label_column].nunique()} classes")
    print(f"   Sampled: {len(df_sampled):,} samples, {df_sampled[label_column].nunique()} classes")
    print(f"   Reduction: {(1 - len(df_sampled)/len(df))*100:.1f}%")
    print(f"   Avg samples/class: {len(df_sampled) / df_sampled[label_column].nunique():.1f}")
    
    return df_sampled


# ============================================================
# 🔧 UPDATED: run_extract_ with Enhanced Centroid-Based Sampling
# ============================================================
def run_extract_(
        df_tax,
        columns_select,
        output_path=None,
        generate_dummy=True,
        sampling_strategy='centroid_closest',
        sample_fraction=0.1,
        min_samples_per_class=10,
        kmer_size=6,
        small_class_threshold=10,  # 🆕 NEW
        small_class_strategy='group',  # 🆕 NEW
        label_used=None,
        sample_per_label=None
        ):
    """
    Extract and save taxonomic data with advanced sampling strategies
    WITH SMALL CLASS HANDLING
    
    Parameters:
    -----------
    df_tax : DataFrame
        Taxonomic data with sequences
    columns_select : list
        List of taxonomic levels to process
    output_path : str
        Output directory path
    generate_dummy : bool
        If True, apply sampling strategy
    sampling_strategy : str
        'centroid_closest', 'centroid_diverse', 'centroid_kmeans', 
        'stratified', 'balanced', 'top_n'
    sample_fraction : float
        Fraction to sample from each class
    min_samples_per_class : int
        Minimum samples per class setelah sampling
    kmer_size : int
        K-mer size for centroid methods
    small_class_threshold : int (NEW)
        Classes with ≤ this many samples are considered "small"
    small_class_strategy : str (NEW)
        How to handle small classes: 'group', 'skip', 'keep'
    label_used : int (deprecated)
        For backward compatibility
    sample_per_label : int (deprecated)
        For backward compatibility
    
    Returns:
    --------
    csv_paths : list
        List of paths to saved CSV files
    paths_file : str
        Path to file containing list of CSV paths
    
    Examples:
    ---------
    # With small class grouping (RECOMMENDED)
    >>> csv_paths, paths_file = run_extract_(
    ...     df_tax,
    ...     columns_select=['genus', 'species'],
    ...     output_path=folders['dataset'],
    ...     generate_dummy=True,
    ...     sampling_strategy='centroid_closest',
    ...     sample_fraction=0.1,
    ...     small_class_threshold=10,
    ...     small_class_strategy='group'  # Group small classes
    ... )
    """
    
    csv_paths = []
    
    for column in columns_select:
        print(f"\n{'='*70}")
        print(f"🔬 Processing level: {column.upper()}")
        print(f"{'='*70}")
        
        # Step 1: Filter awal
        df_result, csv_path = level_extract_plot_freq(
            df_tax, 
            path_=output_path, 
            level=column
        )
        
        # Step 2: Apply sampling strategy
        if generate_dummy:
            print(f"\n📊 Applying sampling strategy: {sampling_strategy}")
            
            # ============================================================
            # CENTROID-BASED METHODS (ENHANCED)
            # ============================================================
            if sampling_strategy.startswith('centroid_'):
                method = sampling_strategy.replace('centroid_', '')
                
                df_sampled = centroid_based_sampling(
                    df=df_result,
                    column_name='sequence',
                    label_column='label',
                    sample_fraction=sample_fraction,
                    min_samples=min_samples_per_class,
                    kmer_size=kmer_size,
                    method=method,
                    small_class_threshold=small_class_threshold,
                    small_class_strategy=small_class_strategy
                )
            
            # ============================================================
            # STRATIFIED RANDOM
            # ============================================================
            elif sampling_strategy == 'stratified':
                print(f"   📦 Stratified random sampling: {sample_fraction*100:.1f}%")
                
                df_sampled = (
                    df_result.groupby('label', group_keys=False)
                    .apply(lambda x: x.sample(
                        n=max(min_samples_per_class, int(len(x) * sample_fraction)),
                        random_state=42,
                        replace=False if len(x) >= min_samples_per_class else True
                    ))
                    .reset_index(drop=True)
                )
                
                print(f"   ✅ Sampled: {len(df_sampled)} samples from {df_sampled['label'].nunique()} classes")
            
            # ============================================================
            # BALANCED
            # ============================================================
            elif sampling_strategy == 'balanced':
                n_per_class = sample_per_label if sample_per_label else min_samples_per_class
                print(f"   ⚖️ Balanced sampling: {n_per_class} samples per class")
                
                df_sampled = (
                    df_result.groupby('label', group_keys=False)
                    .apply(lambda x: x.sample(
                        n=min(len(x), n_per_class),
                        random_state=42
                    ))
                    .reset_index(drop=True)
                )
                
                print(f"   ✅ Sampled: {len(df_sampled)} samples from {df_sampled['label'].nunique()} classes")
            
            # ============================================================
            # TOP_N (Deprecated)
            # ============================================================
            elif sampling_strategy == 'top_n':
                print(f"   ⚠️ Using deprecated 'top_n' strategy")
                
                label_counts = df_result['label'].value_counts()
                n_labels = label_used if label_used else 10
                top_labels = label_counts.head(n_labels).index
                
                df_top = df_result[df_result['label'].isin(top_labels)].copy()
                n_per_class = sample_per_label if sample_per_label else 10
                
                df_sampled = (
                    df_top.groupby('label', group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=42))
                    .reset_index(drop=True)
                )
                
                print(f"   ⚠️ Only {n_labels} top classes used")
                print(f"   ✅ Sampled: {len(df_sampled)} samples from {df_sampled['label'].nunique()} classes")
            
            else:
                raise ValueError(f"Unknown sampling_strategy: '{sampling_strategy}'")
            
            # ============================================================
            # Print Statistics
            # ============================================================
            print(f"\n📊 COMPARISON:")
            print(f"   Before: {len(df_result):,} samples, {df_result['label'].nunique()} classes")
            print(f"   After:  {len(df_sampled):,} samples, {df_sampled['label'].nunique()} classes")
            print(f"   Reduction: {(1 - len(df_sampled)/len(df_result))*100:.1f}%")
            
        else:
            print(f"   📦 Using FULL dataset (no sampling)")
            df_sampled = df_result.copy()
        
        # ============================================================
        # Save to CSV
        # ============================================================
        if output_path:
            folder_path = os.path.join(output_path, column)
            os.makedirs(folder_path, exist_ok=True)
            
            filename = f'{column}.csv' if generate_dummy else f'{column}.csv'
            csv_sampled_path = os.path.join(folder_path, filename)
            
            df_sampled.to_csv(csv_sampled_path, index=False)
            print(f"\n   💾 Saved to: {csv_sampled_path}")
            
            csv_paths.append(csv_sampled_path)
    
    # ============================================================
    # Save Path List
    # ============================================================
    paths_file = None
    if output_path and len(csv_paths) > 0:
        paths_file = os.path.join(output_path, 'csv_level_paths_list.txt')
        with open(paths_file, 'w') as f:
            for path in csv_paths:
                f.write(f"{path}\n")
        print(f"\n📝 Path list saved to: {paths_file}")
    
    return csv_paths, paths_file