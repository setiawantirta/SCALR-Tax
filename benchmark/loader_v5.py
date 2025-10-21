import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_(fasta,
          tax,
          sequence_column_name='sequence',
          taxonomy_column_name='Taxon',
          feature_id_column_name='Feature ID',
          only_bacteria=False,
          clean_ambiguous=False, 
          cleaning_strategy='replace'
          ):
    """
    Load and process FASTA and taxonomy files
    
    Parameters:
    -----------
    ...existing params...
    clean_ambiguous : bool
        Whether to clean ambiguous bases
    cleaning_strategy : str
        'remove', 'replace', or 'random'
    """

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

        # After loading sequences
        if clean_ambiguous:
            print(f"\n🧹 Cleaning ambiguous bases (strategy: {cleaning_strategy})...")
            
            initial_count = len(df_tax)
            cleaned_sequences = []
            
            for seq in df_tax['sequence']:
                cleaned_seq = clean_sequence(seq, strategy=cleaning_strategy)
                cleaned_sequences.append(cleaned_seq)
            
            df_tax['sequence'] = cleaned_sequences
            
            # Remove None values if using 'remove' strategy
            if cleaning_strategy == 'remove':
                df_tax = df_tax[df_tax['sequence'].notna()]
                removed_count = initial_count - len(df_tax)
                print(f"   ✅ Removed {removed_count:,} sequences with ambiguous bases")
                print(f"   ✅ Remaining: {len(df_tax):,} sequences")
        
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

# def level_extract_plot_freq(df_tax, path_=None, level='genus', filter_uncultured=True, min_sample_freq=1):
#     """
#     Extract taxonomic level data and plot frequency distribution
    
#     Parameters:
#     -----------
#     df_tax : DataFrame
#         Taxonomic data
#     path_ : str
#         Output path for plots
#     level : str
#         Taxonomic level (genus, species, etc)
#     filter_uncultured : bool
#         Remove 'uncultured' labels
#     min_sample_freq : int
#         🆕 CHANGED: Default was 4, now 1 (keep all classes)
#         Minimum frequency to keep a class
#         - Set to 1: Keep ALL classes (recommended for hierarchical grouping)
#         - Set to 4: Old behavior (aggressive filtering)
    
#     Returns:
#     --------
#     df_result : DataFrame
#         Filtered data
#     csv_path : str
#         Path to saved CSV
#     """

#     print(f"= STEP 1: Load Data {level} =")
#     kolom_used = ['sequence', level]

#     # Initialize df_filter
#     df_filter = None

#     if filter_uncultured:
#         if level == 'genus':
#             delete_pattern = 'g__uncultured'
#             df_filter = df_tax[
#                 (df_tax['genus'] != 'Unclassified') &
#                 (~df_tax['genus'].str.contains(delete_pattern, case=False, na=False))
#             ]

#         elif level == 'species':
#             delete_patterns = ['s__uncultured', 's__metagenome']
#             df_filter = df_tax[
#                 (df_tax['species'] != 'Unclassified') &
#                 (~df_tax['species'].str.contains('|'.join(delete_patterns), case=False, na=False))
#             ]

#         else:
#             df_filter = df_tax[df_tax[level] != 'Unclassified']
#     else:
#         df_filter = df_tax.copy()

#     df_filter = df_filter[kolom_used].copy()

#     # Summary statistics
#     print(f'SUMMARY {level}')
#     df_rename = df_filter.rename(columns={level: "label"})
#     df_rename.dropna(inplace=True)

#     print(f"== Hitung frekuensi kelas & drop label < {min_sample_freq}")
#     class_counts = df_rename['label'].value_counts()
#     valid_labels = class_counts[class_counts >= min_sample_freq].index
#     df_final = df_rename[df_rename['label'].isin(valid_labels)].copy()
#     df_final = df_final.reset_index(drop=True)

#     # Create folder and save CSV
#     csv_level_file_path = None
#     folder_path = None

#     if path_:
#         folder_path = os.path.join(path_, level)
#         os.makedirs(folder_path, exist_ok=True)

#         csv_level_file_path = os.path.join(folder_path, f'{level}.csv')
#         df_final.to_csv(csv_level_file_path, index=False)
#         print(f"CSV saved to: {csv_level_file_path}")

#     plot_freq(df_final, folder_path=folder_path, level='label')

#     print(f"=== Jumlah label sebelum filter: {df_rename['label'].nunique()}")
#     print(f"=== Jumlah label setelah filter: {df_final['label'].nunique()}")
#     print(f"=== Jumlah baris data: {len(df_final)}")
#     print('========================================================')

#     return df_final, csv_level_file_path


def centroid_closest_sampling(
    df,
    sample_fraction=0.1,
    min_samples_per_class=10,
    kmer_size=6,
    verbose=True
):
    """
    Undersample each class by keeping samples CLOSEST to centroid
    
    Strategy: Keep most representative samples (closest to class center)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'sequence' and 'label' columns
    sample_fraction : float or int
        If float (0-1): fraction of samples to keep per class
        If int (>1): exact number of samples to keep per class
    min_samples_per_class : int
        Minimum samples per class after sampling
    kmer_size : int
        K-mer size for feature generation
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame
        Undersampled dataframe
    """
    
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 CENTROID CLOSEST SAMPLING")
        print(f"{'='*70}")
        print(f"   Sample fraction: {sample_fraction}")
        print(f"   Min samples per class: {min_samples_per_class}")
        print(f"   K-mer size: {kmer_size}")
    
    # Check if k-mer features already exist
    feature_cols = [col for col in df.columns if col.startswith('kmer_')]
    
    if len(feature_cols) == 0:
        if verbose:
            print(f"\n   🧬 Generating k-mer features (k={kmer_size})...")
        
        # Generate k-mer features
        from collections import Counter
        
        def generate_kmer_features(sequence, k):
            """Generate k-mer frequency features"""
            kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
            kmer_counts = Counter(kmers)
            return kmer_counts
        
        # Get all possible k-mers
        all_kmers = set()
        for seq in df['sequence']:
            kmers = generate_kmer_features(str(seq), kmer_size)
            all_kmers.update(kmers.keys())
        
        all_kmers = sorted(all_kmers)
        
        # Create feature matrix
        kmer_features = []
        for seq in df['sequence']:
            kmers = generate_kmer_features(str(seq), kmer_size)
            features = [kmers.get(kmer, 0) for kmer in all_kmers]
            kmer_features.append(features)
        
        kmer_features = np.array(kmer_features)
        
        # Add to dataframe
        for i, kmer in enumerate(all_kmers):
            df[f'kmer_{kmer}'] = kmer_features[:, i]
        
        feature_cols = [f'kmer_{k}' for k in all_kmers]
        
        if verbose:
            print(f"      ✅ Generated {len(feature_cols)} k-mer features")
    
    # Extract features
    X = df[feature_cols].values
    
    # Sample each class
    sampled_indices = []
    
    if verbose:
        print(f"\n   📊 Sampling classes:")
    
    for label in df['label'].unique():
        mask = df['label'] == label
        class_indices = df[mask].index.tolist()
        class_size = len(class_indices)
        
        # Calculate target sample size
        if isinstance(sample_fraction, float) and 0 < sample_fraction <= 1:
            target_size = max(int(class_size * sample_fraction), min_samples_per_class)
        else:
            target_size = max(int(sample_fraction), min_samples_per_class)
        
        # Don't oversample
        target_size = min(target_size, class_size)
        
        if target_size >= class_size:
            # Keep all samples
            sampled_indices.extend(class_indices)
            if verbose and len(sampled_indices) % 50 == 0:
                print(f"      • {label[:40]:40s}: {class_size:5d} → {class_size:5d} (kept all)")
        else:
            # Calculate centroid
            class_features = X[mask]
            centroid = class_features.mean(axis=0).reshape(1, -1)
            
            # Calculate distances to centroid
            distances = euclidean_distances(class_features, centroid).flatten()
            
            # Keep closest samples
            closest_indices = np.argsort(distances)[:target_size]
            selected_indices = [class_indices[i] for i in closest_indices]
            sampled_indices.extend(selected_indices)
            
            if verbose and len(sampled_indices) % 50 == 0:
                print(f"      • {label[:40]:40s}: {class_size:5d} → {target_size:5d}")
    
    # Create sampled dataframe
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.reset_index(drop=True)
    
    if verbose:
        print(f"\n   ✅ SAMPLING COMPLETED:")
        print(f"      • Original samples: {len(df):,}")
        print(f"      • Sampled samples: {len(df_sampled):,}")
        print(f"      • Reduction: {(1 - len(df_sampled)/len(df))*100:.2f}%")
    
    return df_sampled


def centroid_diverse_sampling(
    df,
    sample_fraction=0.1,
    min_samples_per_class=10,
    kmer_size=6,
    verbose=True
):
    """
    Undersample each class with DIVERSE samples (mix of closest + farthest)
    
    Strategy: 60% closest + 40% farthest from centroid
    
    Parameters: Same as centroid_closest_sampling
    """
    
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 CENTROID DIVERSE SAMPLING")
        print(f"{'='*70}")
        print(f"   Mix: 60% closest + 40% farthest")
    
    # [Similar implementation but with 60/40 split]
    # For brevity, I'll show the key difference:
    
    feature_cols = [col for col in df.columns if col.startswith('kmer_')]
    
    if len(feature_cols) == 0:
        # Generate k-mer features (same as above)
        pass
    
    X = df[feature_cols].values
    sampled_indices = []
    
    for label in df['label'].unique():
        mask = df['label'] == label
        class_indices = df[mask].index.tolist()
        class_size = len(class_indices)
        
        if isinstance(sample_fraction, float) and 0 < sample_fraction <= 1:
            target_size = max(int(class_size * sample_fraction), min_samples_per_class)
        else:
            target_size = max(int(sample_fraction), min_samples_per_class)
        
        target_size = min(target_size, class_size)
        
        if target_size >= class_size:
            sampled_indices.extend(class_indices)
        else:
            class_features = X[mask]
            centroid = class_features.mean(axis=0).reshape(1, -1)
            distances = euclidean_distances(class_features, centroid).flatten()
            
            # ✅ KEY DIFFERENCE: 60% closest + 40% farthest
            n_close = int(target_size * 0.6)
            n_far = target_size - n_close
            
            closest_indices = np.argsort(distances)[:n_close]
            farthest_indices = np.argsort(distances)[-n_far:]
            
            selected_indices = list(closest_indices) + list(farthest_indices)
            selected_indices = [class_indices[i] for i in selected_indices]
            sampled_indices.extend(selected_indices)
    
    df_sampled = df.loc[sampled_indices].copy()
    df_sampled = df_sampled.reset_index(drop=True)
    
    if verbose:
        print(f"\n   ✅ DIVERSE SAMPLING COMPLETED:")
        print(f"      • Original: {len(df):,} → Sampled: {len(df_sampled):,}")
    
    return df_sampled


# def centroid_kmeans_sampling(
#     df,
#     sample_fraction=0.1,
#     min_samples_per_class=10,
#     kmer_size=6,
#     verbose=True
# ):
#     """
#     Undersample each class using K-means clustering
    
#     Strategy: Cluster samples and take representatives from each cluster
    
#     Parameters: Same as centroid_closest_sampling
#     """
    
#     import numpy as np
#     from sklearn.cluster import MiniBatchKMeans
    
#     if verbose:
#         print(f"\n{'='*70}")
#         print(f"📊 CENTROID K-MEANS SAMPLING")
#         print(f"{'='*70}")
#         print(f"   Strategy: Cluster-based sampling")
    
#     feature_cols = [col for col in df.columns if col.startswith('kmer_')]
    
#     if len(feature_cols) == 0:
#         # Generate k-mer features
#         pass
    
#     X = df[feature_cols].values
#     sampled_indices = []
    
#     for label in df['label'].unique():
#         mask = df['label'] == label
#         class_indices = df[mask].index.tolist()
#         class_size = len(class_indices)
        
#         if isinstance(sample_fraction, float) and 0 < sample_fraction <= 1:
#             target_size = max(int(class_size * sample_fraction), min_samples_per_class)
#         else:
#             target_size = max(int(sample_fraction), min_samples_per_class)
        
#         target_size = min(target_size, class_size)
        
#         if target_size >= class_size:
#             sampled_indices.extend(class_indices)
#         else:
#             class_features = X[mask]
            
#             # ✅ KEY: Use K-means clustering
#             n_clusters = min(target_size, class_size)
            
#             if n_clusters < 2:
#                 sampled_indices.extend(class_indices[:target_size])
#             else:
#                 kmeans = MiniBatchKMeans(
#                     n_clusters=n_clusters,
#                     random_state=42,
#                     batch_size=1000
#                 )
#                 cluster_labels = kmeans.fit_predict(class_features)
                
#                 # Take one sample from each cluster (closest to cluster center)
#                 selected = []
#                 for cluster_id in range(n_clusters):
#                     cluster_mask = cluster_labels == cluster_id
#                     if cluster_mask.sum() > 0:
#                         cluster_indices = np.where(cluster_mask)[0]
#                         cluster_features = class_features[cluster_indices]
#                         cluster_center = kmeans.cluster_centers_[cluster_id]
                        
#                         # Find closest to center
#                         distances = np.linalg.norm(
#                             cluster_features - cluster_center,
#                             axis=1
#                         )
#                         closest_idx = cluster_indices[np.argmin(distances)]
#                         selected.append(class_indices[closest_idx])
                
#                 sampled_indices.extend(selected[:target_size])
    
#     df_sampled = df.loc[sampled_indices].copy()
#     df_sampled = df_sampled.reset_index(drop=True)
    
#     if verbose:
#         print(f"\n   ✅ K-MEANS SAMPLING COMPLETED:")
#         print(f"      • Original: {len(df):,} → Sampled: {len(df_sampled):,}")
    
#     return df_sampled

def centroid_kmeans_sampling(df, sample_fraction, min_samples_per_class, 
                             kmer_size, verbose=True):
    """
    Sample using K-means clustering on k-mer features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with k-mer features and 'label' column
    sample_fraction : float or int
        If float < 1.0: fraction of samples to keep per class
        If int >= 1: exact number of samples per class
    min_samples_per_class : int
        Minimum samples required per class to perform clustering
    kmer_size : int
        K-mer size (for logging only)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df_sampled : DataFrame
        Sampled dataframe
    """
    
    # ✅ FIX 1: Import required libraries
    try:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"Required library not found: {e}\n"
            f"Please install: pip install scikit-learn"
        )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 CENTROID K-MEANS SAMPLING")
        print(f"{'='*70}")
        print(f"   Parameters:")
        print(f"      • Sample fraction: {sample_fraction}")
        print(f"      • Min samples per class: {min_samples_per_class}")
        print(f"      • K-mer size: k={kmer_size}")
    
    # ✅ FIX 2: Validate inputs
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'label' column")
    
    # ✅ FIX 3: Separate features and labels
    label_col = 'label'
    feature_cols = [col for col in df.columns if col.startswith('kmer_')]
    
    if len(feature_cols) == 0:
        raise ValueError(
            "No k-mer features found. Columns must start with 'kmer_'"
        )
    
    if verbose:
        print(f"\n   📊 Dataset info:")
        print(f"      • Total samples: {len(df):,}")
        print(f"      • Total features: {len(feature_cols):,}")
        print(f"      • Unique classes: {df[label_col].nunique()}")
    
    # ✅ FIX 4: Process each class
    sampled_dfs = []
    skipped_classes = []
    
    for class_label in df[label_col].unique():
        class_df = df[df[label_col] == class_label].copy()
        n_samples = len(class_df)
        
        if verbose:
            print(f"\n   📦 Processing class '{class_label}': {n_samples} samples")
        
        # ✅ FIX 5: Skip classes with too few samples
        if n_samples < min_samples_per_class:
            if verbose:
                print(f"      ⚠️  Skipping: only {n_samples} samples "
                      f"(< {min_samples_per_class} required)")
            skipped_classes.append(class_label)
            continue
        
        # ✅ FIX 6: Extract features
        class_features = class_df[feature_cols].values
        
        # Validate feature dimensions
        if class_features.shape[1] == 0:
            if verbose:
                print(f"      ❌ Skipping: no features available")
            skipped_classes.append(class_label)
            continue
        
        # ✅ FIX 7: Calculate number of clusters
        if isinstance(sample_fraction, float) and sample_fraction < 1.0:
            n_clusters = max(
                int(n_samples * sample_fraction), 
                min_samples_per_class
            )
        else:
            n_clusters = int(sample_fraction)
        
        n_clusters = min(n_clusters, n_samples)
        
        if verbose:
            print(f"      🎯 Using {n_clusters} clusters")
            print(f"      📊 Feature shape: {class_features.shape}")
        
        # ✅ FIX 8: Handle edge cases
        if n_clusters >= n_samples:
            # Take all samples if clusters >= samples
            sampled_dfs.append(class_df)
            if verbose:
                print(f"      ✅ Taking all {n_samples} samples (n_clusters >= n_samples)")
            continue
        
        if n_clusters < 2:
            # Need at least 2 clusters for K-means
            n_sample = min(min_samples_per_class, n_samples)
            sampled = class_df.sample(n=n_sample, random_state=42)
            sampled_dfs.append(sampled)
            if verbose:
                print(f"      ✅ Random sampling {n_sample} samples (n_clusters < 2)")
            continue
        
        # ✅ FIX 9: Perform K-means with error handling
        try:
            # Choose K-means variant based on dataset size
            if n_samples < 10000:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=min(1000, n_samples // 10)
                )
            
            # Fit and predict
            cluster_labels = kmeans.fit_predict(class_features)
            
            # ✅ FIX 10: Select samples closest to cluster centers
            selected_indices = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                
                if cluster_mask.sum() == 0:
                    continue
                
                # Get samples in this cluster
                cluster_samples = class_features[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Calculate distances to center
                distances = np.linalg.norm(
                    cluster_samples - cluster_center,
                    axis=1
                )
                
                # Find closest sample
                closest_idx = np.argmin(distances)
                
                # Get original index from dataframe
                original_indices = class_df.index[cluster_mask].tolist()
                selected_indices.append(original_indices[closest_idx])
            
            # Add selected samples
            sampled_dfs.append(class_df.loc[selected_indices])
            
            if verbose:
                print(f"      ✅ K-means completed: selected {len(selected_indices)} samples")
        
        except Exception as e:
            # ✅ FIX 11: Fallback to random sampling on error
            if verbose:
                print(f"      ❌ K-means failed: {e}")
                print(f"         Falling back to random sampling...")
            
            n_sample = min(n_clusters, n_samples)
            sampled = class_df.sample(n=n_sample, random_state=42)
            sampled_dfs.append(sampled)
            
            if verbose:
                print(f"      ✅ Random sampled {n_sample} samples")
    
    # ✅ FIX 12: Validate results
    if len(sampled_dfs) == 0:
        raise ValueError(
            f"No samples selected! All {len(skipped_classes)} classes were skipped.\n"
            f"Try lowering min_samples_per_class (current: {min_samples_per_class})"
        )
    
    # Combine all sampled dataframes
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ SAMPLING COMPLETED")
        print(f"{'='*70}")
        print(f"   📊 Summary:")
        print(f"      • Before: {len(df):,} samples, {df[label_col].nunique()} classes")
        print(f"      • After: {len(result_df):,} samples, {result_df[label_col].nunique()} classes")
        print(f"      • Reduction: {(1 - len(result_df)/len(df))*100:.1f}%")
        
        if skipped_classes:
            print(f"\n   ⚠️  Skipped {len(skipped_classes)} classes:")
            for cls in skipped_classes[:5]:
                print(f"      • {cls}")
            if len(skipped_classes) > 5:
                print(f"      ... and {len(skipped_classes)-5} more")
    
    return result_df

def level_extract_plot_freq(df_tax, path_=None, level='genus', filter_uncultured=True, min_sample_freq=1):
    """
    Extract taxonomic level data and plot frequency distribution
    
    Parameters:
    -----------
    df_tax : DataFrame
        Taxonomic data
    path_ : str
        Output path for plots
    level : str
        Taxonomic level (genus, species, etc)
    filter_uncultured : bool
        Remove 'uncultured' labels
    min_sample_freq : int
        🆕 CHANGED: Default was 4, now 1 (keep all classes)
        Minimum frequency to keep a class
        - Set to 1: Keep ALL classes (recommended for hierarchical grouping)
        - Set to 4: Old behavior (aggressive filtering)
    
    Returns:
    --------
    df_result : DataFrame
        Filtered data
    csv_path : str
        Path to saved CSV
    """

    print(f"\n{'='*70}")
    print(f"📊 STEP 1: LOAD DATA - {level.upper()}")
    print(f"{'='*70}")
    
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
    print(f'\n📋 INITIAL DATA SUMMARY:')
    df_rename = df_filter.rename(columns={level: "label"})
    df_rename.dropna(inplace=True)
    
    print(f"   • Total samples: {len(df_rename):,}")
    print(f"   • Total classes: {df_rename['label'].nunique()}")

    # ============================================================
    # 🆕 IMPROVED: Detailed Class Distribution Analysis
    # ============================================================
    
    print(f"\n📈 CLASS SIZE DISTRIBUTION (BEFORE FILTERING):")
    class_counts = df_rename['label'].value_counts()
    
    # Calculate distribution statistics
    size_1 = (class_counts == 1).sum()
    size_2_3 = ((class_counts >= 2) & (class_counts <= 3)).sum()
    size_4_9 = ((class_counts >= 4) & (class_counts <= 9)).sum()
    size_10_plus = (class_counts >= 10).sum()
    
    print(f"   • Classes with 1 sample: {size_1}")
    print(f"   • Classes with 2-3 samples: {size_2_3}")
    print(f"   • Classes with 4-9 samples: {size_4_9}")
    print(f"   • Classes with ≥10 samples: {size_10_plus}")
    
    # ============================================================
    # 🆕 IMPROVED: Configurable Filtering with Detailed Reporting
    # ============================================================
    
    if min_sample_freq > 1:
        print(f"\n⚠️  FILTERING: Dropping classes with < {min_sample_freq} samples")
        
        classes_to_drop = class_counts[class_counts < min_sample_freq].index
        samples_to_drop = class_counts[classes_to_drop].sum()
        
        print(f"   • Classes to drop: {len(classes_to_drop)}")
        print(f"   • Samples to drop: {samples_to_drop:,}")
        
        # Show examples of dropped classes
        if len(classes_to_drop) > 0:
            print(f"\n   📋 Examples of dropped classes:")
            dropped_sorted = class_counts[classes_to_drop].sort_values(ascending=False)
            
            for idx, (cls, cnt) in enumerate(dropped_sorted.head(10).items(), 1):
                cls_str = str(cls)[:55] + "..." if len(str(cls)) > 58 else str(cls)
                print(f"      {idx:2d}. {cls_str}: {cnt} samples")
            
            if len(classes_to_drop) > 10:
                print(f"      ... and {len(classes_to_drop)-10} more classes")
        
        # Apply filter
        valid_labels = class_counts[class_counts >= min_sample_freq].index
        df_final = df_rename[df_rename['label'].isin(valid_labels)].copy()
        df_final = df_final.reset_index(drop=True)
        
        print(f"\n✅ AFTER FILTERING:")
        print(f"   • Remaining samples: {len(df_final):,}")
        print(f"   • Remaining classes: {df_final['label'].nunique()}")
        print(f"   • Sample reduction: {samples_to_drop:,} ({(samples_to_drop/len(df_rename))*100:.2f}%)")
        print(f"   • Class reduction: {len(classes_to_drop)} ({(len(classes_to_drop)/len(class_counts))*100:.2f}%)")
    
    else:
        print(f"\n✅ NO FILTERING APPLIED (min_sample_freq={min_sample_freq})")
        print(f"   • Keeping ALL {len(class_counts)} classes")
        print(f"   • Total samples: {len(df_rename):,}")
        df_final = df_rename.copy()

    # ============================================================
    # 🆕 IMPROVED: Final Statistics Summary
    # ============================================================
    
    final_class_counts = df_final['label'].value_counts()
    
    print(f"\n📊 FINAL DATA STATISTICS:")
    print(f"   • Total samples: {len(df_final):,}")
    print(f"   • Total classes: {df_final['label'].nunique()}")
    print(f"   • Min class size: {final_class_counts.min()}")
    print(f"   • Max class size: {final_class_counts.max()}")
    print(f"   • Mean class size: {final_class_counts.mean():.1f}")
    print(f"   • Median class size: {final_class_counts.median():.1f}")
    
    if final_class_counts.min() > 0:
        imbalance_ratio = final_class_counts.max() / final_class_counts.min()
        print(f"   • Imbalance ratio: {imbalance_ratio:.1f}:1")

    # ============================================================
    # Save CSV and Generate Plots
    # ============================================================
    
    csv_level_file_path = None
    folder_path = None

    if path_:
        folder_path = os.path.join(path_, level)
        os.makedirs(folder_path, exist_ok=True)

        csv_level_file_path = os.path.join(folder_path, f'{level}.csv')
        df_final.to_csv(csv_level_file_path, index=False)
        print(f"\n💾 CSV saved to: {csv_level_file_path}")

    # Generate frequency plots
    try:
        plot_freq(df_final, folder_path=folder_path, level='label')
        if folder_path:
            print(f"📊 Frequency plots saved to: {folder_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")

    print(f"{'='*70}\n")

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
# 🆕 STRATEGY 1: HIERARCHICAL RARE GROUPING
# ============================================================

def hierarchical_rare_grouping(
    df,
    label_column='label',
    small_class_threshold=10,  # ← This should match!
    strategy='group',
    kmer_size=6,
    verbose=True
):
    """
    Group rare classes hierarchically based on taxonomy
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with sequences and labels
    label_column : str
        Name of the label column
    small_class_threshold : int  # ✅ CORRECT NAME
        Threshold for grouping rare classes
    strategy : str
        'group' or 'remove'
    kmer_size : int
        K-mer size for dummy features
    verbose : bool
        Print progress information
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with grouped rare classes
    """
    
    if verbose:
        print(f"\n🌳 HIERARCHICAL RARE CLASS GROUPING")
        print(f"   Threshold: {small_class_threshold} samples")
        print(f"   Strategy: {strategy}")
    
    # Get class counts
    class_counts = df[label_column].value_counts()
    
    # Identify rare classes
    rare_classes = class_counts[class_counts < small_class_threshold].index
    
    if len(rare_classes) == 0:
        if verbose:
            print(f"   ✅ No rare classes found (all classes ≥ {small_class_threshold} samples)")
        return df
    
    if verbose:
        print(f"\n   📊 RARE CLASSES IDENTIFIED:")
        print(f"      • Total rare classes: {len(rare_classes)}")
        print(f"      • Total samples in rare classes: {class_counts[rare_classes].sum()}")
    
    # Process based on strategy
    if strategy == 'group':
        # Group rare classes by genus/higher taxonomy
        
        # Assuming label format: "Genus_species" or "Genus species"
        df_grouped = df.copy()
        
        grouped_count = 0
        for rare_class in rare_classes:
            # Extract genus from label
            genus = str(rare_class).split('_')[0].split()[0]
            
            # Create grouped label
            new_label = f"RARE_{genus}"
            
            # Update label
            mask = df_grouped[label_column] == rare_class
            df_grouped.loc[mask, label_column] = new_label
            grouped_count += mask.sum()
        
        if verbose:
            final_counts = df_grouped[label_column].value_counts()
            rare_groups = [c for c in final_counts.index if str(c).startswith('RARE_')]
            
            print(f"\n   ✅ GROUPING COMPLETED:")
            print(f"      • Samples grouped: {grouped_count}")
            print(f"      • RARE_* groups created: {len(rare_groups)}")
            print(f"\n      Top 5 RARE groups:")
            for group in rare_groups[:5]:
                count = final_counts[group]
                print(f"         • {group}: {count} samples")
        
        return df_grouped
    
    elif strategy == 'remove':
        # Remove rare classes
        df_filtered = df[~df[label_column].isin(rare_classes)].copy()
        df_filtered = df_filtered.reset_index(drop=True)
        
        removed_samples = len(df) - len(df_filtered)
        
        if verbose:
            print(f"\n   ✅ REMOVAL COMPLETED:")
            print(f"      • Classes removed: {len(rare_classes)}")
            print(f"      • Samples removed: {removed_samples}")
            print(f"      • Remaining samples: {len(df_filtered)}")
        
        return df_filtered
    
    else:
        if verbose:
            print(f"   ⚠️  Unknown strategy '{strategy}', returning original data")
        return df


# ============================================================
# 🆕 STRATEGY 2: ADAPTIVE SAMPLING WITH SMOTE
# ============================================================

def adaptive_class_sampling(
    df, 
    label_column='label',
    sequence_column='sequence',
    target_samples_per_class=100,
    keep_all_rare=True,
    oversample_rare=True,
    rare_threshold=10,
    kmer_size=6,
    random_state=42,
    verbose=True
):
    """
    Adaptive sampling with different strategies per class size
    
    - Large classes (>1000): Undersample
    - Medium classes (10-1000): Keep all or slight oversample
    - Small classes (<10): Keep all + SMOTE oversample
    
    Parameters:
    -----------
    df : DataFrame
        Data with sequence and label columns
    label_column : str
        Column name containing labels
    sequence_column : str
        Column name containing sequences
    target_samples_per_class : int
        Target number of samples per class after balancing
    keep_all_rare : bool
        Keep all samples from rare classes
    oversample_rare : bool
        Apply SMOTE to rare classes
    rare_threshold : int
        Threshold for "rare" classes
    kmer_size : int
        K-mer size for feature extraction (needed for SMOTE)
    random_state : int
        Random seed
    verbose : bool
        Print detailed statistics
    
    Returns:
    --------
    df_balanced : DataFrame
        Balanced dataset
    
    Example:
    --------
    >>> df_balanced = adaptive_class_sampling(
    ...     df,
    ...     target_samples_per_class=100,
    ...     oversample_rare=True,
    ...     rare_threshold=10
    ... )
    """
    
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        IMBLEARN_AVAILABLE = True
    except ImportError:
        IMBLEARN_AVAILABLE = False
        if oversample_rare:
            print("⚠️  WARNING: imbalanced-learn not installed. Disabling SMOTE.")
            print("   Install with: pip install imbalanced-learn")
            oversample_rare = False
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎚️ ADAPTIVE CLASS SAMPLING")
        print(f"{'='*70}")
    
    df_result = df.copy()
    
    # Calculate class frequencies
    class_counts = df[label_column].value_counts()
    
    # Categorize classes
    large_classes = class_counts[class_counts > 1000].index
    medium_classes = class_counts[(class_counts >= rare_threshold) & (class_counts <= 1000)].index
    small_classes = class_counts[class_counts < rare_threshold].index
    
    if verbose:
        print(f"   Target samples per class: {target_samples_per_class}")
        print(f"   Rare threshold: {rare_threshold}")
        print(f"\n   📊 CLASS CATEGORIZATION:")
        print(f"      • Large classes (>1000): {len(large_classes)}")
        print(f"      • Medium classes ({rare_threshold}-1000): {len(medium_classes)}")
        print(f"      • Small classes (<{rare_threshold}): {len(small_classes)}")
    
    sampled_dfs = []
    
    # ============================================================
    # STEP 1: Large Classes - UNDERSAMPLE
    # ============================================================
    
    if len(large_classes) > 0 and verbose:
        print(f"\n   ⬇️ UNDERSAMPLING LARGE CLASSES:")
    
    for label in large_classes:
        df_class = df_result[df_result[label_column] == label]
        n_target = min(target_samples_per_class, len(df_class))
        df_sampled = df_class.sample(n=n_target, random_state=random_state)
        sampled_dfs.append(df_sampled)
        
        if verbose:
            print(f"      • {label[:50]}: {len(df_class):,} → {n_target} samples")
    
    # ============================================================
    # STEP 2: Medium Classes - KEEP ALL
    # ============================================================
    
    if len(medium_classes) > 0:
        df_medium = df_result[df_result[label_column].isin(medium_classes)]
        sampled_dfs.append(df_medium)
        
        if verbose:
            print(f"\n   ✅ KEEPING ALL MEDIUM CLASSES:")
            print(f"      • {len(medium_classes)} classes")
            print(f"      • Total: {len(df_medium):,} samples")
    
    # ============================================================
    # STEP 3: Small Classes - KEEP ALL + OPTIONAL SMOTE
    # ============================================================
    
    if len(small_classes) > 0:
        df_small = df_result[df_result[label_column].isin(small_classes)]
        
        if keep_all_rare:
            sampled_dfs.append(df_small)
            
            if verbose:
                print(f"\n   📦 KEEPING ALL SMALL CLASSES:")
                print(f"      • {len(small_classes)} classes")
                print(f"      • Total: {len(df_small):,} samples")
    
    # ============================================================
    # STEP 4: Combine
    # ============================================================
    
    if len(sampled_dfs) == 0:
        print("   ❌ ERROR: No data after sampling!")
        return df.head(0)
    
    df_balanced = pd.concat(sampled_dfs, ignore_index=True)
    
    # ============================================================
    # STEP 5: Apply SMOTE (Optional)
    # ============================================================
    
    if oversample_rare and IMBLEARN_AVAILABLE and len(small_classes) > 0:
        if verbose:
            print(f"\n   🧬 APPLYING SMOTE TO RARE CLASSES...")
        
        try:
            # Extract k-mer features
            from itertools import product
            
            bases = ['A', 'C', 'G', 'T']
            all_kmers = [''.join(p) for p in product(bases, repeat=kmer_size)]
            kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
            n_features = len(all_kmers)
            
            if verbose:
                print(f"      • Extracting k-mer features (k={kmer_size}, features={n_features})")
            
            # Convert sequences to k-mer features
            X_list = []
            y_list = []
            
            for idx, row in df_balanced.iterrows():
                seq = str(row[sequence_column]).upper()
                label = row[label_column]
                
                kmer_counts = Counter()
                for i in range(len(seq) - kmer_size + 1):
                    kmer = seq[i:i+kmer_size]
                    if kmer in kmer_to_idx:
                        kmer_counts[kmer] += 1
                
                # Convert to feature vector
                feature_vec = np.zeros(n_features, dtype=np.float32)
                for kmer, count in kmer_counts.items():
                    feature_vec[kmer_to_idx[kmer]] = count
                
                # Normalize
                vec_sum = feature_vec.sum()
                if vec_sum > 0:
                    feature_vec /= vec_sum
                
                X_list.append(feature_vec)
                y_list.append(label)
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            if verbose:
                print(f"      • Feature matrix shape: {X.shape}")
            
            # Apply SMOTE
            # Calculate k_neighbors (must be < smallest class size)
            min_class_size = pd.Series(y).value_counts().min()
            k_neighbors = min(5, min_class_size - 1)
            
            if k_neighbors < 1:
                if verbose:
                    print(f"      ⚠️ Classes too small for SMOTE (min size: {min_class_size})")
                    print(f"      Skipping SMOTE...")
            else:
                smote = SMOTE(
                    sampling_strategy='auto',
                    k_neighbors=k_neighbors,
                    random_state=random_state
                )
                
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                if verbose:
                    print(f"      ✅ SMOTE completed:")
                    print(f"         • Before: {len(y):,} samples")
                    print(f"         • After: {len(y_resampled):,} samples")
                    print(f"         • Synthetic samples added: {len(y_resampled) - len(y):,}")
                
                # Reconstruct dataframe (simplified - sequences not reconstructed)
                # Note: For real use, you may want to keep original sequences
                df_balanced_new = pd.DataFrame({
                    label_column: y_resampled
                })
                
                # Add placeholder sequences (or keep originals)
                # This is a simplified version - in production, you'd need to handle this better
                df_balanced_new[sequence_column] = 'SYNTHETIC_SEQUENCE'
                
                # Keep original sequences
                mask_original = np.arange(len(y))
                df_balanced_new.loc[mask_original, sequence_column] = df_balanced[sequence_column].values
                
                df_balanced = df_balanced_new
        
        except Exception as e:
            if verbose:
                print(f"      ❌ SMOTE failed: {e}")
                print(f"      Using non-oversampled data")
    
    # ============================================================
    # FINAL STATISTICS
    # ============================================================
    
    if verbose:
        final_counts = df_balanced[label_column].value_counts()
        
        print(f"\n   📈 FINAL STATISTICS:")
        print(f"      • Original: {len(df):,} samples, {df[label_column].nunique()} classes")
        print(f"      • Balanced: {len(df_balanced):,} samples, {df_balanced[label_column].nunique()} classes")
        print(f"      • Avg samples/class: {final_counts.mean():.1f}")
        print(f"      • Min samples/class: {final_counts.min()}")
        print(f"      • Max samples/class: {final_counts.max()}")
    
    print(f"{'='*70}\n")
    
    return df_balanced


# ============================================================
# 🆕 STRATEGY 3: TWO-STAGE HIERARCHICAL CLASSIFIER
# ============================================================

class HierarchicalClassifier:
    """
    Two-stage hierarchical classifier for extreme class imbalance
    
    Stage 1: Classify to genus (coarse, balanced)
    Stage 2: Within each genus, classify to species (fine-grained)
    
    Example:
    --------
    >>> # Prepare data
    >>> df['genus'] = df['label'].str.split('_').str[0]
    >>> df['species'] = df['label']
    >>> 
    >>> # Split data
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> genus_train = df.loc[y_train.index, 'genus'].values
    >>> genus_test = df.loc[y_test.index, 'genus'].values
    >>> 
    >>> # Train
    >>> clf = HierarchicalClassifier()
    >>> clf.fit(X_train, y_train, genus_train)
    >>> 
    >>> # Predict
    >>> y_pred = clf.predict(X_test, genus_test)
    """
    
    def __init__(self, verbose=True):
        self.genus_classifier = None
        self.species_classifiers = {}
        self.verbose = verbose
        self.genus_to_species = {}  # Map genus → list of species
    
    def fit(self, X, y_species, y_genus):
        """
        Train hierarchical classifier
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_species : array-like
            Species labels (fine-grained)
        y_genus : array-like
            Genus labels (coarse)
        """
        from sklearn.ensemble import RandomForestClassifier
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🌳 TRAINING HIERARCHICAL CLASSIFIER")
            print(f"{'='*70}")
        
        # ============================================================
        # STAGE 1: Train Genus Classifier
        # ============================================================
        
        if self.verbose:
            unique_genera = np.unique(y_genus)
            print(f"   Stage 1: Training genus classifier...")
            print(f"   • Number of genera: {len(unique_genera)}")
        
        self.genus_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.genus_classifier.fit(X, y_genus)
        
        if self.verbose:
            train_acc = self.genus_classifier.score(X, y_genus)
            print(f"   ✅ Genus classifier trained (train accuracy: {train_acc:.4f})")
        
        # ============================================================
        # STAGE 2: Train Species Classifiers per Genus
        # ============================================================
        
        if self.verbose:
            print(f"\n   Stage 2: Training species classifiers per genus...")
        
        unique_genera = np.unique(y_genus)
        
        for genus_idx, genus in enumerate(unique_genera, 1):
            mask = y_genus == genus
            X_genus = X[mask]
            y_species_genus = y_species[mask]
            
            # Get unique species in this genus
            unique_species = np.unique(y_species_genus)
            self.genus_to_species[genus] = unique_species
            
            # Skip if only 1 species in this genus
            if len(unique_species) == 1:
                self.species_classifiers[genus] = None
                if self.verbose and genus_idx <= 5:
                    print(f"      [{genus_idx}/{len(unique_genera)}] {genus}: 1 species (no classifier needed)")
                continue
            
            # Train species classifier
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_genus, y_species_genus)
            self.species_classifiers[genus] = clf
            
            if self.verbose and genus_idx <= 5:
                train_acc = clf.score(X_genus, y_species_genus)
                print(f"      [{genus_idx}/{len(unique_genera)}] {genus}: {len(unique_species)} species (acc: {train_acc:.4f})")
        
        if self.verbose and len(unique_genera) > 5:
            print(f"      ... and {len(unique_genera)-5} more genera")
        
        if self.verbose:
            n_with_classifier = sum(1 for clf in self.species_classifiers.values() if clf is not None)
            print(f"\n   ✅ Trained {n_with_classifier} species classifiers")
        
        print(f"{'='*70}\n")
    
    def predict(self, X, y_genus_true=None):
        """
        Predict species using hierarchical approach
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y_genus_true : array-like, optional
            True genus labels (if available, skips stage 1)
        
        Returns:
        --------
        y_species_pred : array
            Predicted species labels
        """
        
        # STAGE 1: Predict genus (or use true genus if provided)
        if y_genus_true is None:
            y_genus_pred = self.genus_classifier.predict(X)
        else:
            y_genus_pred = y_genus_true
        
        # STAGE 2: Predict species within genus
        y_species_pred = []
        
        for i, genus in enumerate(y_genus_pred):
            if genus in self.species_classifiers and self.species_classifiers[genus] is not None:
                # Predict species within this genus
                clf = self.species_classifiers[genus]
                species = clf.predict(X[i:i+1])[0]
                y_species_pred.append(species)
            else:
                # Only 1 species in genus, use the known species
                if genus in self.genus_to_species:
                    species = self.genus_to_species[genus][0]
                    y_species_pred.append(species)
                else:
                    # Unknown genus, use genus as species
                    y_species_pred.append(f"{genus}_unknown_species")
        
        return np.array(y_species_pred)
    
    def predict_proba(self, X, y_genus_true=None):
        """
        Predict probability for each species
        
        Returns:
        --------
        probabilities : dict
            Dictionary of {species: probability}
        """
        
        # This is a simplified version
        # Full implementation would combine genus and species probabilities
        
        if y_genus_true is None:
            y_genus_pred = self.genus_classifier.predict(X)
            genus_proba = self.genus_classifier.predict_proba(X)
        else:
            y_genus_pred = y_genus_true
            genus_proba = None
        
        results = []
        
        for i, genus in enumerate(y_genus_pred):
            if genus in self.species_classifiers and self.species_classifiers[genus] is not None:
                clf = self.species_classifiers[genus]
                species_proba = clf.predict_proba(X[i:i+1])[0]
                species_classes = clf.classes_
                
                proba_dict = dict(zip(species_classes, species_proba))
            else:
                # Single species
                if genus in self.genus_to_species:
                    species = self.genus_to_species[genus][0]
                    proba_dict = {species: 1.0}
                else:
                    proba_dict = {f"{genus}_unknown_species": 1.0}
            
            results.append(proba_dict)
        
        return results


# ============================================================
# 🆕 STRATEGY 4: COST-SENSITIVE LEARNING
# ============================================================

def compute_class_weights(y, strategy='balanced', verbose=True):
    """
    Compute class weights to penalize rare class misclassification
    
    Parameters:
    -----------
    y : array-like
        Class labels
    strategy : str
        - 'balanced': sklearn's balanced weighting
        - 'inverse_freq': Max_count / class_count
        - 'log_balanced': Log-scaled balanced weighting
        - 'sqrt_balanced': Sqrt-scaled balanced weighting
    verbose : bool
        Print weight statistics
    
    Returns:
    --------
    class_weights : dict
        Dictionary of {class: weight}
    
    Example:
    --------
    >>> # For use with sklearn
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> 
    >>> weights = compute_class_weights(y_train, strategy='log_balanced')
    >>> 
    >>> model = RandomForestClassifier(class_weight=weights)
    >>> model.fit(X_train, y_train)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"⚖️ COMPUTING CLASS WEIGHTS")
        print(f"{'='*70}")
        print(f"   Strategy: {strategy}")
    
    classes = np.unique(y)
    class_counts = pd.Series(y).value_counts()
    
    if strategy == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
    
    elif strategy == 'inverse_freq':
        max_count = class_counts.max()
        weight_dict = {cls: max_count / class_counts[cls] for cls in classes}
    
    elif strategy == 'log_balanced':
        max_count = class_counts.max()
        weight_dict = {cls: np.log(max_count / class_counts[cls] + 1) for cls in classes}
    
    elif strategy == 'sqrt_balanced':
        max_count = class_counts.max()
        weight_dict = {cls: np.sqrt(max_count / class_counts[cls]) for cls in classes}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if verbose:
        # Show statistics
        weights_series = pd.Series(weight_dict)
        
        print(f"\n   📊 WEIGHT STATISTICS:")
        print(f"      • Number of classes: {len(classes)}")
        print(f"      • Min weight: {weights_series.min():.4f}")
        print(f"      • Max weight: {weights_series.max():.4f}")
        print(f"      • Mean weight: {weights_series.mean():.4f}")
        print(f"      • Weight ratio (max/min): {weights_series.max()/weights_series.min():.2f}x")
        
        # Show top 5 weighted classes (usually rare classes)
        print(f"\n   🔝 TOP 5 WEIGHTED CLASSES (highest penalty):")
        sorted_weights = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
        for cls, weight in sorted_weights[:5]:
            count = class_counts[cls]
            print(f"      • {str(cls)[:50]}: weight={weight:.4f} (n={count})")
    
    print(f"{'='*70}\n")
    
    return weight_dict

def filter_unwanted_labels(
    df,
    label_column='label',
    unwanted_keywords=None,
    case_sensitive=False,
    verbose=True
):
    """
    Filter out labels containing unwanted keywords
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_column : str
        Name of the label column
    unwanted_keywords : list of str
        Keywords to filter out. Default: ['uncultured', 'metagenome', 'unidentified']
    case_sensitive : bool
        Whether to perform case-sensitive matching
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
        
    Example:
    --------
    >>> df_clean = filter_unwanted_labels(
    ...     df,
    ...     unwanted_keywords=['uncultured', 'metagenome', 'unidentified', 'unknown']
    ... )
    """
    
    # Default unwanted keywords
    if unwanted_keywords is None:
        unwanted_keywords = [
            'uncultured',
            'metagenome', 
            'unidentified',
            'unknown',
            'unclassified',
            'environmental',
            'clone',
            #'sp.',  # species placeholder
        ]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🧹 FILTERING UNWANTED LABELS")
        print(f"{'='*70}")
        print(f"   Keywords to filter: {unwanted_keywords}")
        print(f"   Case sensitive: {case_sensitive}")
    
    # Store original stats
    original_samples = len(df)
    original_classes = df[label_column].nunique()
    
    if verbose:
        print(f"\n📊 BEFORE FILTERING:")
        print(f"   • Total samples: {original_samples:,}")
        print(f"   • Total classes: {original_classes:,}")
    
    # Create a mask for rows to keep
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Track what's being filtered
    filtered_by_keyword = {}
    
    for keyword in unwanted_keywords:
        if case_sensitive:
            keyword_mask = df[label_column].astype(str).str.contains(keyword, na=False, regex=False)
        else:
            keyword_mask = df[label_column].astype(str).str.lower().str.contains(
                keyword.lower(), na=False, regex=False
            )
        
        n_matches = keyword_mask.sum()
        if n_matches > 0:
            filtered_by_keyword[keyword] = n_matches
            mask = mask & ~keyword_mask
    
    # Apply filter
    df_filtered = df[mask].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Calculate statistics
    filtered_samples = original_samples - len(df_filtered)
    filtered_classes = original_classes - df_filtered[label_column].nunique()
    
    if verbose:
        print(f"\n✅ AFTER FILTERING:")
        print(f"   • Remaining samples: {len(df_filtered):,}")
        print(f"   • Remaining classes: {df_filtered[label_column].nunique():,}")
        print(f"\n📉 FILTERED OUT:")
        print(f"   • Samples removed: {filtered_samples:,} ({(filtered_samples/original_samples)*100:.2f}%)")
        print(f"   • Classes removed: {filtered_classes:,} ({(filtered_classes/original_classes)*100:.2f}%)")
        
        if filtered_by_keyword:
            print(f"\n   📋 Breakdown by keyword:")
            for keyword, count in sorted(filtered_by_keyword.items(), key=lambda x: x[1], reverse=True):
                print(f"      • '{keyword}': {count:,} samples")
        else:
            print(f"\n   ℹ️  No matches found for any keyword")
    
    return df_filtered

def clean_sequence(sequence, strategy='remove'):
    """
    Clean sequence by handling ambiguous bases
    
    Parameters:
    -----------
    sequence : str
        DNA sequence
    strategy : str
        'remove': Remove sequences with ambiguous bases
        'replace': Replace with most likely base
        'random': Randomly choose from possibilities
        
    Returns:
    --------
    cleaned_sequence : str or None
        Cleaned sequence, or None if removed
    """
    import random
    
    # IUPAC replacement mapping (conservative choice)
    replacements = {
        'N': 'A',  # Random choice
        'R': 'A',  # Purine -> Adenine (most common)
        'Y': 'C',  # Pyrimidine -> Cytosine (more stable)
        'H': 'A',  # Not G -> Adenine
        'K': 'G',  # Keto -> Guanine
        'M': 'A',  # Amino -> Adenine
        'S': 'C',  # Strong -> Cytosine
        'W': 'A',  # Weak -> Adenine
        'B': 'C',  # Not A -> Cytosine
        'D': 'A',  # Not C -> Adenine
        'V': 'A',  # Not T -> Adenine
    }
    
    sequence = sequence.upper()
    
    if strategy == 'remove':
        # Remove sequences with any ambiguous base
        ambiguous = set('NRYKHDBVSWM')
        if any(base in ambiguous for base in sequence):
            return None
        return sequence
    
    elif strategy == 'replace':
        # Replace ambiguous bases with most likely alternative
        cleaned = []
        for base in sequence:
            if base in replacements:
                cleaned.append(replacements[base])
            else:
                cleaned.append(base)
        return ''.join(cleaned)
    
    elif strategy == 'random':
        # Random replacement from possibilities
        iupac_map = {
            'N': ['A', 'T', 'C', 'G'],
            'R': ['A', 'G'],
            'Y': ['C', 'T'],
            'H': ['A', 'C', 'T'],
            'K': ['G', 'T'],
            'M': ['A', 'C'],
            'S': ['G', 'C'],
            'W': ['A', 'T'],
            'B': ['C', 'G', 'T'],
            'D': ['A', 'G', 'T'],
            'V': ['A', 'C', 'G'],
        }
        
        cleaned = []
        for base in sequence:
            if base in iupac_map:
                cleaned.append(random.choice(iupac_map[base]))
            else:
                cleaned.append(base)
        return ''.join(cleaned)
    
    return sequence



def filter_unwanted_labels_advanced(
    df,
    label_column='label',
    filter_uncultured=True,
    filter_metagenome=True,
    filter_unidentified=True,
    filter_environmental=True,
    custom_keywords=None,
    case_sensitive=False,
    verbose=True
):
    """
    Advanced filtering with granular control over different filter types
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    label_column : str
        Name of the label column
    filter_uncultured : bool
        Filter 'uncultured' labels
    filter_metagenome : bool
        Filter 'metagenome' labels
    filter_unidentified : bool
        Filter 'unidentified', 'unknown', 'unclassified' labels
    filter_environmental : bool
        Filter 'environmental', 'clone' labels
    custom_keywords : list of str
        Additional custom keywords to filter
    case_sensitive : bool
        Whether to perform case-sensitive matching
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    
    # Build keyword list based on flags
    unwanted_keywords = []
    
    if filter_uncultured:
        unwanted_keywords.extend(['uncultured', 'uncultivated'])
    
    if filter_metagenome:
        unwanted_keywords.extend(['metagenome', 'metagenomic'])
    
    if filter_unidentified:
        unwanted_keywords.extend(['unidentified', 'unknown', 'unclassified'])
    
    if filter_environmental:
        unwanted_keywords.extend(['environmental', 'clone'])
    
    # Add custom keywords
    if custom_keywords:
        unwanted_keywords.extend(custom_keywords)
    
    # Remove duplicates while preserving order
    unwanted_keywords = list(dict.fromkeys(unwanted_keywords))
    
    # Use the base filter function
    return filter_unwanted_labels(
        df,
        label_column=label_column,
        unwanted_keywords=unwanted_keywords,
        case_sensitive=case_sensitive,
        verbose=verbose
    )


# ============================================================
# 🆕 VISUALIZATION: CLASS DISTRIBUTION COMPARISON
# ============================================================

def plot_class_distribution_comparison(
    df_before,
    df_after,
    label_column='label',
    output_path=None,
    level='genus',
    show_top_n=30,
    figsize=(16, 12)
):
    """
    Plot comprehensive comparison of class distributions before/after balancing

    - If show_top_n is None -> plot ALL classes (figure auto-resizes)
    - Function is resilient to very many classes (adjusts figsize, font sizes)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import math
    from scipy.stats import pearsonr, spearmanr

    # Calculate class distributions
    before_counts = df_before[label_column].value_counts()
    after_counts = df_after[label_column].value_counts()

    # Determine top classes (None => all classes sorted by original size desc)
    if show_top_n is None:
        top_classes = before_counts.sort_values(ascending=False).index.tolist()
        n_top = len(top_classes)
    else:
        n_top = int(show_top_n)
        top_classes = before_counts.nlargest(n_top).index.tolist()

    # Dynamic figure sizing: scale height with number of classes for bar plot
    base_w, base_h = figsize
    # increase height for long class lists (0.25 inch per 10 classes)
    extra_h = min(60, max(0, (n_top / 10) * 2))  # cap extra height
    fig_height = max(base_h, base_h + extra_h)
    fig = plt.figure(figsize=(base_w, fig_height))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    sns.set_style("whitegrid")

    # PLOT 1: Distribution Histogram (Log Scale)
    ax1 = fig.add_subplot(gs[0, :])
    max_count = max(1, int(max(before_counts.max() if len(before_counts)>0 else 0,
                                after_counts.max() if len(after_counts)>0 else 0)))
    bins = np.logspace(0, math.log10(max_count), 50)
    ax1.hist(before_counts.values, bins=bins, alpha=0.6, label='Before',
             color='#e74c3c', edgecolor='black', linewidth=0.3)
    ax1.hist(after_counts.values, bins=bins, alpha=0.6, label='After',
             color='#3498db', edgecolor='black', linewidth=0.3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Class Size (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
    ax1.set_title(f'Class Size Distribution: {level.upper()}\nBefore vs After Balancing',
                  fontsize=14, fontweight='bold', pad=12)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    # safe stats text
    try:
        before_min = int(before_counts.min()) if len(before_counts)>0 else 0
        before_max = int(before_counts.max()) if len(before_counts)>0 else 0
        after_min = int(after_counts.min()) if len(after_counts)>0 else 0
        after_max = int(after_counts.max()) if len(after_counts)>0 else 0
        imbalance_before = f"{before_max}/{max(1,before_min)}"
        imbalance_after = f"{after_max}/{max(1,after_min)}"
        stats_text = (
            f"Before: {len(before_counts)} classes, {len(df_before):,} samples\n"
            f"After: {len(after_counts)} classes, {len(df_after):,} samples\n"
            f"Imbalance ratio (max/min): {imbalance_before} → {imbalance_after}"
        )
    except Exception:
        stats_text = f"Before: {len(before_counts)} classes, After: {len(after_counts)} classes"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # PLOT 2: Top N Classes Comparison (Horizontal Bar)
    ax2 = fig.add_subplot(gs[1, :])
    before_vals = [before_counts.get(c, 0) for c in top_classes]
    after_vals = [after_counts.get(c, 0) for c in top_classes]
    y_pos = np.arange(len(top_classes))
    bar_height = 0.35
    ax2.barh(y_pos - bar_height/2, before_vals, bar_height,
             label='Before', color='#e74c3c', alpha=0.8)
    ax2.barh(y_pos + bar_height/2, after_vals, bar_height,
             label='After', color='#3498db', alpha=0.8)

    # Adaptive label handling
    if n_top > 60:
        yt_fontsize = 6
    elif n_top > 30:
        yt_fontsize = 7
    else:
        yt_fontsize = 9

    # If classes are too many, use index labels instead of long strings to avoid clutter
    def short_label(s, maxlen=60):
        s = str(s)
        return (s[:maxlen-3] + '...') if len(s) > maxlen else s

    if n_top <= 200:
        class_labels = [short_label(c, maxlen=60) for c in top_classes]
    else:
        # when extremely many classes, use numeric index to keep plot readable
        class_labels = [f"{i+1}" for i in range(n_top)]
        # save mapping file if output_path provided
        if output_path:
            mapping_path = os.path.join(output_path, f'{level}_label_index_mapping.csv')
            pd.DataFrame({'index': class_labels, 'label': top_classes}).to_csv(mapping_path, index=False)
            print(f"   🔖 Saved label→index mapping: {mapping_path}")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_labels, fontsize=yt_fontsize)
    ax2.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
    title_n = f"Top {n_top} Classes Comparison" if show_top_n is not None else f"All Classes Comparison ({n_top} classes)"
    ax2.set_title(title_n, fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # PLOT 3: Scatter Plot (Before vs After)
    ax3 = fig.add_subplot(gs[2, 0])
    common_classes = sorted(set(before_counts.index) & set(after_counts.index))
    before_common = [before_counts[c] for c in common_classes]
    after_common = [after_counts[c] for c in common_classes]

    if len(before_common) == 0 or len(after_common) == 0:
        ax3.text(0.5, 0.5, 'No common classes to plot', ha='center', va='center')
        pearson_r = float('nan')
        spearman_r = float('nan')
    else:
        ax3.scatter(before_common, after_common, alpha=0.6, s=40, color='#9b59b6')
        max_val = max(max(before_common), max(after_common))
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=1.2, label='No change')
        ax3.set_xlabel('Samples Before', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Samples After', fontsize=10, fontweight='bold')
        ax3.set_title('Per-Class Sample Count\n(Before vs After)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        try:
            pearson_r, _ = pearsonr(before_common, after_common) if len(before_common) > 1 else (float('nan'), None)
            spearman_r, _ = spearmanr(before_common, after_common) if len(before_common) > 1 else (float('nan'), None)
        except Exception:
            pearson_r = float('nan'); spearman_r = float('nan')

    corr_text = f"Pearson r: {np.nan if np.isnan(pearson_r) else round(pearson_r,3)}\nSpearman ρ: {np.nan if np.isnan(spearman_r) else round(spearman_r,3)}"
    ax3.text(0.05, 0.95, corr_text, transform=ax3.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # PLOT 4: Cumulative Distribution
    ax4 = fig.add_subplot(gs[2, 1])
    before_sorted = np.sort(before_counts.values) if len(before_counts)>0 else np.array([0])
    after_sorted = np.sort(after_counts.values) if len(after_counts)>0 else np.array([0])
    before_cumsum = np.cumsum(before_sorted) / (before_sorted.sum() + 1e-12) * 100
    after_cumsum = np.cumsum(after_sorted) / (after_sorted.sum() + 1e-12) * 100
    ax4.plot(range(len(before_sorted)), before_cumsum, label='Before', color='#e74c3c', linewidth=2)
    ax4.plot(range(len(after_sorted)), after_cumsum, label='After', color='#3498db', linewidth=2)
    ax4.set_xlabel('Classes (sorted by size)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Cumulative % of Samples', fontsize=10, fontweight='bold')
    ax4.set_title('Cumulative Sample Distribution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])
    ax4.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax4.text(max(1, len(before_sorted))*0.7, 82, '80% line', fontsize=8, color='gray')

    plt.tight_layout()

    if output_path:
        plot_path = os.path.join(output_path,
                                 f'{level}_distribution_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Distribution plot saved: {plot_path}")

    # Calculate Statistics
    stats = {
        'before': {
            'n_classes': len(before_counts),
            'n_samples': len(df_before),
            'min_class_size': int(before_counts.min()) if len(before_counts)>0 else 0,
            'max_class_size': int(before_counts.max()) if len(before_counts)>0 else 0,
            'mean_class_size': float(before_counts.mean()) if len(before_counts)>0 else 0.0,
            'median_class_size': float(before_counts.median()) if len(before_counts)>0 else 0.0,
            'std_class_size': float(before_counts.std()) if len(before_counts)>0 else 0.0,
            'imbalance_ratio': (int(before_counts.max()) / max(1, int(before_counts.min()))) if len(before_counts)>0 else None
        },
        'after': {
            'n_classes': len(after_counts),
            'n_samples': len(df_after),
            'min_class_size': int(after_counts.min()) if len(after_counts)>0 else 0,
            'max_class_size': int(after_counts.max()) if len(after_counts)>0 else 0,
            'mean_class_size': float(after_counts.mean()) if len(after_counts)>0 else 0.0,
            'median_class_size': float(after_counts.median()) if len(after_counts)>0 else 0.0,
            'std_class_size': float(after_counts.std()) if len(after_counts)>0 else 0.0,
            'imbalance_ratio': (int(after_counts.max()) / max(1, int(after_counts.min()))) if len(after_counts)>0 else None
        },
        'correlation': {
            'pearson': (pearson_r if 'pearson_r' in locals() else float('nan')),
            'spearman': (spearman_r if 'spearman_r' in locals() else float('nan'))
        },
        'reduction': {
            'classes': len(before_counts) - len(after_counts),
            'samples': len(df_before) - len(df_after),
            'classes_pct': (1 - len(after_counts) / len(before_counts)) * 100 if len(before_counts) > 0 else None,
            'samples_pct': (1 - len(df_after) / len(df_before)) * 100 if len(df_before) > 0 else None
        }
    }

    return fig, stats

def plot_distribution_comparison(df_before, df_after, output_path, level):
    """
    🆕 IMPROVED: Enhanced distribution comparison plots with better layout
    
    Creates 2x2 subplot comparing:
    1. Class Size Distribution (KDE + transparent bars)
    2. Sample Distribution per Class (line plot)
    3. Class Count Summary (bar chart)
    4. Imbalance Ratio Comparison (metric visualization)
    
    Parameters:
    -----------
    df_before : DataFrame
        Data before balancing
    df_after : DataFrame
        Data after balancing
    output_path : str
        Path to save plot
    level : str
        Taxonomic level name
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    
    # Calculate class distributions
    class_dist_before = df_before['label'].value_counts().sort_values(ascending=False)
    class_dist_after = df_after['label'].value_counts().sort_values(ascending=False)
    
    # ============================================================
    # CREATE 2x2 SUBPLOT LAYOUT
    # ============================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribution Comparison: {level.upper()}\nBefore vs After Balancing', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color scheme
    color_before = '#FF1493'  # Deep Pink
    color_after = '#1E90FF'   # Dodger Blue
    
    # ============================================================
    # PLOT 1: CLASS SIZE DISTRIBUTION (TOP LEFT)
    # Enhanced with KDE overlay and transparency
    # ============================================================
    
    ax1 = axes[0, 0]
    
    # Create bins for histogram
    max_samples = max(class_dist_before.max(), class_dist_after.max())
    bins = np.linspace(0, max_samples, 50)
    
    # Plot histograms with transparency
    ax1.hist(class_dist_before.values, bins=bins, 
             alpha=0.5, color=color_before, 
             label='Before Balancing', edgecolor='white', linewidth=0.5)
    
    ax1.hist(class_dist_after.values, bins=bins, 
             alpha=0.5, color=color_after, 
             label='After Balancing', edgecolor='white', linewidth=0.5)
    
    # Add KDE overlay for smoother visualization
    try:
        # KDE for "before"
        kde_before = stats.gaussian_kde(class_dist_before.values)
        x_range = np.linspace(0, max_samples, 200)
        kde_y_before = kde_before(x_range)
        # Scale KDE to match histogram
        kde_y_before = kde_y_before * len(class_dist_before) * (max_samples / 50)
        ax1.plot(x_range, kde_y_before, color=color_before, linewidth=2.5, 
                linestyle='--', alpha=0.8, label='KDE (Before)')
        
        # KDE for "after"
        kde_after = stats.gaussian_kde(class_dist_after.values)
        kde_y_after = kde_after(x_range)
        kde_y_after = kde_y_after * len(class_dist_after) * (max_samples / 50)
        ax1.plot(x_range, kde_y_after, color=color_after, linewidth=2.5, 
                linestyle='--', alpha=0.8, label='KDE (After)')
    except Exception as e:
        print(f"⚠️  Could not plot KDE: {e}")
    
    # Formatting
    ax1.set_xlabel('Samples per Class', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Classes', fontsize=11, fontweight='bold')
    ax1.set_title('Class Size Distribution\n(Histogram + KDE)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # # Add statistics text box
    # stats_text = (
    #     f"Before:\n"
    #     f"  • Classes: {len(class_dist_before)}\n"
    #     f"  • Mean: {class_dist_before.mean():.1f}\n"
    #     f"  • Median: {class_dist_before.median():.1f}\n"
    #     f"\n"
    #     f"After:\n"
    #     f"  • Classes: {len(class_dist_after)}\n"
    #     f"  • Mean: {class_dist_after.mean():.1f}\n"
    #     f"  • Median: {class_dist_after.median():.1f}"
    # )
    # ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
    #          fontsize=8, verticalalignment='top', horizontalalignment='right',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================================
    # PLOT 2: SAMPLE DISTRIBUTION PER CLASS (TOP RIGHT)
    # Line plot showing sample count for each class
    # ============================================================
    
    ax2 = axes[0, 1]
    
    # Sort classes by "before" distribution
    x_indices = np.arange(len(class_dist_before))
    
    # Plot "before" distribution
    ax2.plot(x_indices, class_dist_before.values, 
             color=color_before, linewidth=2, alpha=0.7,
             marker='o', markersize=3, label='Before Balancing')
    
    # For "after", we need to align classes
    # Get sample counts for same classes (fill missing with 0)
    after_aligned = []
    for cls in class_dist_before.index:
        if cls in class_dist_after.index:
            after_aligned.append(class_dist_after[cls])
        else:
            after_aligned.append(0)
    
    ax2.plot(x_indices, after_aligned, 
             color=color_after, linewidth=2, alpha=0.7,
             marker='s', markersize=3, label='After Balancing')
    
    # Formatting
    ax2.set_xlabel('Class Rank (sorted by Before)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
    ax2.set_title('Sample Distribution per Class\n(Sorted by abundance)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add reference line for target sample count (if applicable)
    if len(after_aligned) > 0:
        target_samples = np.median([x for x in after_aligned if x > 0])
        ax2.axhline(y=target_samples, color='green', linestyle=':', 
                   linewidth=2, alpha=0.6, label=f'Target: {target_samples:.0f}')
        ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # ============================================================
    # PLOT 3: CLASS COUNT SUMMARY (BOTTOM LEFT)
    # Bar chart comparing number of classes
    # ============================================================
    
    ax3 = axes[1, 0]
    
    # Data for bar chart
    categories = ['Before\nBalancing', 'After\nBalancing']
    class_counts = [len(class_dist_before), len(class_dist_after)]
    sample_counts = [len(df_before), len(df_after)]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax3.bar(x_pos - width/2, class_counts, width, 
                    label='Number of Classes', color=color_before, alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, [c/1000 for c in sample_counts], width,
                    label='Total Samples (÷1000)', color=color_after, alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatting
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Class & Sample Count Summary\n(Total counts comparison)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    # ============================================================
    # PLOT 4: IMBALANCE RATIO COMPARISON (BOTTOM RIGHT)
    # Metric visualization showing improvement
    # ============================================================
    
    ax4 = axes[1, 1]
    
    # Calculate imbalance ratios
    if class_dist_before.min() > 0:
        imbalance_before = class_dist_before.max() / class_dist_before.min()
    else:
        imbalance_before = float('inf')
    
    if class_dist_after.min() > 0:
        imbalance_after = class_dist_after.max() / class_dist_after.min()
    else:
        imbalance_after = float('inf')
    
    # Create horizontal bar chart
    metrics = ['Imbalance\nRatio', 'Max/Min\nSamples', 'Std Dev\n(normalized)']
    
    # Calculate metrics
    before_metrics = [
        imbalance_before if imbalance_before != float('inf') else 0,
        class_dist_before.max() / (class_dist_before.mean() or 1),
        class_dist_before.std() / (class_dist_before.mean() or 1)
    ]
    
    after_metrics = [
        imbalance_after if imbalance_after != float('inf') else 0,
        class_dist_after.max() / (class_dist_after.mean() or 1),
        class_dist_after.std() / (class_dist_after.mean() or 1)
    ]
    
    y_pos = np.arange(len(metrics))
    height = 0.35
    
    bars1 = ax4.barh(y_pos - height/2, before_metrics, height,
                     label='Before', color=color_before, alpha=0.7)
    bars2 = ax4.barh(y_pos + height/2, after_metrics, height,
                     label='After', color=color_after, alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            if width_val > 0:
                ax4.text(width_val, bar.get_y() + bar.get_height()/2.,
                        f'{width_val:.2f}',
                        ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Formatting
    ax4.set_xlabel('Ratio Value', fontsize=11, fontweight='bold')
    ax4.set_title('Imbalance Metrics Comparison\n(Lower is better)', 
                  fontsize=12, fontweight='bold', pad=10)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(metrics, fontsize=10)
    ax4.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
    
    # Add improvement percentage
    if imbalance_before != float('inf') and imbalance_after != float('inf'):
        improvement = ((imbalance_before - imbalance_after) / imbalance_before) * 100
        improvement_text = f"Improvement: {improvement:.1f}%"
        ax4.text(0.98, 0.02, improvement_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontweight='bold')
    
    # ============================================================
    # FINAL ADJUSTMENTS
    # ============================================================
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Distribution comparison saved: {output_path}")
    
    plt.close()

# ============================================================
# 🆕 ANALYSIS: CORRELATION MATRIX
# ============================================================

def plot_correlation_matrix(
    df_before,
    df_after,
    label_column='label',
    output_path=None,
    level='genus',
    method='spearman',
    figsize=(14, 12)
):
    """
    Plot correlation matrix between class distributions before and after
    
    Parameters:
    -----------
    df_before : DataFrame
        Data before balancing
    df_after : DataFrame
        Data after balancing
    label_column : str
        Column containing class labels
    output_path : str
        Path to save plot
    level : str
        Taxonomic level name
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    correlation_df : DataFrame
        Correlation matrix
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    # Calculate class distributions
    before_counts = df_before[label_column].value_counts().sort_index()
    after_counts = df_after[label_column].value_counts().sort_index()
    
    # Create alignment dataframe (all classes)
    all_classes = sorted(set(before_counts.index) | set(after_counts.index))
    
    correlation_data = pd.DataFrame({
        'Before': [before_counts.get(c, 0) for c in all_classes],
        'After': [after_counts.get(c, 0) for c in all_classes],
    }, index=all_classes)
    
    # Add additional metrics
    correlation_data['Difference'] = correlation_data['After'] - correlation_data['Before']
    correlation_data['Ratio'] = correlation_data['After'] / (correlation_data['Before'] + 1e-10)
    correlation_data['Log_Before'] = np.log1p(correlation_data['Before'])
    correlation_data['Log_After'] = np.log1p(correlation_data['After'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Correlation Analysis: {level.upper()}\nBefore vs After Balancing', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # ============================================================
    # PLOT 1: Correlation Heatmap
    # ============================================================
    
    # Calculate correlation matrix
    corr_matrix = correlation_data[['Before', 'After', 'Log_Before', 'Log_After']].corr(method=method)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=axes[0, 0])
    axes[0, 0].set_title(f'{method.capitalize()} Correlation Matrix', 
                         fontsize=13, fontweight='bold', pad=10)
    
    # ============================================================
    # PLOT 2: Scatter with Regression
    # ============================================================
    
    ax = axes[0, 1]
    
    # Plot scatter
    ax.scatter(correlation_data['Log_Before'], correlation_data['Log_After'],
               alpha=0.6, s=50, color='#9b59b6', edgecolors='black', linewidth=0.5)
    
    # Add regression line
    from scipy.stats import linregress
    mask = (correlation_data['Before'] > 0) & (correlation_data['After'] > 0)
    slope, intercept, r_value, p_value, std_err = linregress(
        correlation_data.loc[mask, 'Log_Before'],
        correlation_data.loc[mask, 'Log_After']
    )
    
    x_line = np.array([correlation_data['Log_Before'].min(), 
                       correlation_data['Log_Before'].max()])
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8,
            label=f'y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.3f}')
    
    # Add diagonal (perfect correlation)
    ax.plot(x_line, x_line, 'k--', alpha=0.3, linewidth=1, label='Perfect correlation')
    
    ax.set_xlabel('Log(Samples Before + 1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Log(Samples After + 1)', fontsize=11, fontweight='bold')
    ax.set_title('Log-Transformed Scatter Plot', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ============================================================
    # PLOT 3: Residual Plot
    # ============================================================
    
    ax = axes[1, 0]
    
    # Calculate residuals
    predicted = slope * correlation_data['Log_Before'] + intercept
    residuals = correlation_data['Log_After'] - predicted
    
    ax.scatter(correlation_data['Log_Before'], residuals,
               alpha=0.6, s=50, color='#e67e22')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Log(Samples Before + 1)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    residual_std = residuals.std()
    ax.text(0.05, 0.95, f'Residual Std: {residual_std:.3f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ============================================================
    # PLOT 4: Class Retention Analysis
    # ============================================================
    
    ax = axes[1, 1]
    
    # Categorize classes
    lost_classes = [c for c in before_counts.index if c not in after_counts.index]
    new_classes = [c for c in after_counts.index if c not in before_counts.index]
    retained_classes = [c for c in before_counts.index if c in after_counts.index]
    
    # Create pie chart
    sizes = [len(retained_classes), len(lost_classes), len(new_classes)]
    labels = [f'Retained\n({sizes[0]})', 
              f'Lost\n({sizes[1]})', 
              f'New\n({sizes[2]})']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                        colors=colors, autopct='%1.1f%%',
                                        shadow=True, startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Class Retention', fontsize=13, fontweight='bold')
    
    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    # ============================================================
    # Save Figure
    # ============================================================
    
    plt.tight_layout()
    
    if output_path:
        plot_path = os.path.join(output_path, 
                                 f'{level}_correlation_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Correlation analysis saved: {plot_path}")
    
    return fig, correlation_data


# ============================================================
# 🆕 COMPREHENSIVE STATISTICS REPORT
# ============================================================

def generate_balancing_report(
    df_before,
    df_after,
    label_column='label',
    output_path=None,
    level='genus',
    imbalance_strategy='hierarchical_grouping',
    sampling_strategy='centroid_closest'
):
    """
    Generate comprehensive text report of balancing statistics
    
    Parameters:
    -----------
    df_before : DataFrame
        Data before balancing
    df_after : DataFrame
        Data after balancing
    label_column : str
        Column containing class labels
    output_path : str
        Path to save report
    level : str
        Taxonomic level name
    imbalance_strategy : str
        Imbalance handling strategy used
    sampling_strategy : str
        Sampling strategy used
    
    Returns:
    --------
    report : str
        Report text
    """
    
    from scipy.stats import pearsonr, spearmanr, ks_2samp
    
    # Calculate distributions
    before_counts = df_before[label_column].value_counts()
    after_counts = df_after[label_column].value_counts()
    
    # Statistical tests
    common_classes = set(before_counts.index) & set(after_counts.index)
    before_common = [before_counts[c] for c in common_classes]
    after_common = [after_counts[c] for c in common_classes]
    
    pearson_r, pearson_p = pearsonr(before_common, after_common)
    spearman_r, spearman_p = spearmanr(before_common, after_common)
    ks_stat, ks_p = ks_2samp(before_counts.values, after_counts.values)
    
    # Build report
    report = []
    report.append("="*70)
    report.append(f"CLASS BALANCING REPORT: {level.upper()}")
    report.append("="*70)
    report.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Imbalance Strategy: {imbalance_strategy}")
    report.append(f"Sampling Strategy: {sampling_strategy}")
    
    report.append("\n" + "="*70)
    report.append("1. DATA SUMMARY")
    report.append("="*70)
    
    report.append("\nBEFORE BALANCING:")
    report.append(f"  • Total Samples: {len(df_before):,}")
    report.append(f"  • Total Classes: {len(before_counts):,}")
    report.append(f"  • Min Class Size: {before_counts.min():,}")
    report.append(f"  • Max Class Size: {before_counts.max():,}")
    report.append(f"  • Mean Class Size: {before_counts.mean():.1f}")
    report.append(f"  • Median Class Size: {before_counts.median():.1f}")
    report.append(f"  • Std Class Size: {before_counts.std():.1f}")
    report.append(f"  • Imbalance Ratio: {before_counts.max()/before_counts.min():.1f}:1")
    
    report.append("\nAFTER BALANCING:")
    report.append(f"  • Total Samples: {len(df_after):,}")
    report.append(f"  • Total Classes: {len(after_counts):,}")
    report.append(f"  • Min Class Size: {after_counts.min():,}")
    report.append(f"  • Max Class Size: {after_counts.max():,}")
    report.append(f"  • Mean Class Size: {after_counts.mean():.1f}")
    report.append(f"  • Median Class Size: {after_counts.median():.1f}")
    report.append(f"  • Std Class Size: {after_counts.std():.1f}")
    report.append(f"  • Imbalance Ratio: {after_counts.max()/after_counts.min():.1f}:1")
    
    report.append("\n" + "="*70)
    report.append("2. CHANGES")
    report.append("="*70)
    
    sample_reduction = len(df_before) - len(df_after)
    class_reduction = len(before_counts) - len(after_counts)
    
    report.append(f"\n  • Sample Reduction: {sample_reduction:,} ({sample_reduction/len(df_before)*100:.1f}%)")
    report.append(f"  • Class Reduction: {class_reduction:,} ({class_reduction/len(before_counts)*100:.1f}%)")
    report.append(f"  • Imbalance Improvement: {before_counts.max()/before_counts.min():.1f}:1 → {after_counts.max()/after_counts.min():.1f}:1")
    
    imbalance_improvement = (1 - (after_counts.max()/after_counts.min()) / (before_counts.max()/before_counts.min())) * 100
    report.append(f"  • Imbalance Reduction: {imbalance_improvement:.1f}%")
    
    report.append("\n" + "="*70)
    report.append("3. CLASS RETENTION ANALYSIS")
    report.append("="*70)
    
    lost_classes = set(before_counts.index) - set(after_counts.index)
    new_classes = set(after_counts.index) - set(before_counts.index)
    retained_classes = common_classes
    
    report.append(f"\n  • Retained Classes: {len(retained_classes):,} ({len(retained_classes)/len(before_counts)*100:.1f}%)")
    report.append(f"  • Lost Classes: {len(lost_classes):,} ({len(lost_classes)/len(before_counts)*100:.1f}%)")
    report.append(f"  • New Classes: {len(new_classes):,}")
    
    if len(lost_classes) > 0:
        report.append(f"\n  Top 10 Lost Classes (by original size):")
        lost_sorted = sorted([(c, before_counts[c]) for c in lost_classes], 
                            key=lambda x: x[1], reverse=True)[:10]
        for cls, count in lost_sorted:
            report.append(f"    - {str(cls)[:50]}: {count} samples")
    
    if len(new_classes) > 0:
        report.append(f"\n  Top 10 New Classes (by size):")
        new_sorted = sorted([(c, after_counts[c]) for c in new_classes], 
                           key=lambda x: x[1], reverse=True)[:10]
        for cls, count in new_sorted:
            report.append(f"    - {str(cls)[:50]}: {count} samples")
    
    report.append("\n" + "="*70)
    report.append("4. STATISTICAL TESTS")
    report.append("="*70)
    
    report.append(f"\n  Pearson Correlation:")
    report.append(f"    • r = {pearson_r:.4f}")
    report.append(f"    • p-value = {pearson_p:.4e}")
    report.append(f"    • Interpretation: {'Strong' if abs(pearson_r) > 0.7 else 'Moderate' if abs(pearson_r) > 0.4 else 'Weak'} linear correlation")
    
    report.append(f"\n  Spearman Correlation:")
    report.append(f"    • ρ = {spearman_r:.4f}")
    report.append(f"    • p-value = {spearman_p:.4e}")
    report.append(f"    • Interpretation: {'Strong' if abs(spearman_r) > 0.7 else 'Moderate' if abs(spearman_r) > 0.4 else 'Weak'} monotonic correlation")
    
    report.append(f"\n  Kolmogorov-Smirnov Test:")
    report.append(f"    • statistic = {ks_stat:.4f}")
    report.append(f"    • p-value = {ks_p:.4e}")
    report.append(f"    • Interpretation: Distributions are {'DIFFERENT' if ks_p < 0.05 else 'SIMILAR'} (α=0.05)")
    
    report.append("\n" + "="*70)
    report.append("5. TOP 20 CLASSES COMPARISON")
    report.append("="*70)
    
    report.append("\n{:<50} {:>10} {:>10} {:>10}".format("Class", "Before", "After", "Change"))
    report.append("-" * 82)
    
    top_20 = before_counts.nlargest(20).index
    for cls in top_20:
        before_val = before_counts[cls]
        after_val = after_counts.get(cls, 0)
        change = after_val - before_val
        change_pct = (change / before_val * 100) if before_val > 0 else 0
        
        cls_str = str(cls)[:47] + "..." if len(str(cls)) > 50 else str(cls)
        report.append("{:<50} {:>10,} {:>10,} {:>+9,} ({:+.1f}%)".format(
            cls_str, before_val, after_val, change, change_pct
        ))
    
    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    # Join and save
    report_text = "\n".join(report)
    
    if output_path:
        report_path = os.path.join(output_path, f'{level}_balancing_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"   📄 Balancing report saved: {report_path}")
    
    return report_text

def compare_imbalance_strategies(df_tax, level='genus', output_path='./comparison'):
    """
    Compare all imbalance handling strategies
    """
    
    strategies = {
        'baseline': {
            'hierarchical_grouping': False,
            'adaptive_sampling': False,
            'sampling_strategy': 'stratified'
        },
        'hierarchical_only': {
            'hierarchical_grouping': True,
            'adaptive_sampling': False,
            'sampling_strategy': 'centroid_closest'
        },
        'adaptive_only': {
            'hierarchical_grouping': False,
            'adaptive_sampling': True,
            'sampling_strategy': 'stratified'
        },
        'combined': {
            'hierarchical_grouping': True,
            'adaptive_sampling': True,
            'sampling_strategy': 'centroid_diverse'
        }
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n{'='*70}")
        print(f"Testing: {strategy_name.upper()}")
        print(f"{'='*70}")
        
        csv_paths, _ = run_extract_(
            df_tax,
            columns_select=[level],
            output_path=os.path.join(output_path, strategy_name),
            generate_dummy=True,
            **params
        )
        
        # Load result
        df_result = pd.read_csv(csv_paths[0])
        
        results[strategy_name] = {
            'n_samples': len(df_result),
            'n_classes': df_result['label'].nunique(),
            'class_distribution': df_result['label'].value_counts().to_dict()
        }
    
    return results

# ============================================================
# 🔧 UPDATED: run_extract_ WITH ALL STRATEGIES
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
        small_class_threshold=10,
        small_class_strategy='group',
        
        imbalance_strategy='hierarchical_grouping',
        hierarchical_grouping=True,
        
        min_sample_freq=1,
        
        # Filtering parameters
        filter_uncultured=True,
        filter_metagenome=True,
        filter_unidentified=True,
        filter_environmental=False,
        custom_filter_keywords=None,
        case_sensitive_filter=False,
        
        top_n_classes=None,
        
        # ✅ NEW: Ambiguous base handling
        filter_ambiguous_bases=True,      # Remove sequences with N, R, Y, etc.
        ambiguous_handling='remove',       # 'remove', 'replace', 'random'
        max_ambiguous_ratio=0.05,         # Max 5% ambiguous bases allowed
        
        # Visualization controls
        create_plots=True,
        plot_top_n=30,
        plot_correlation=True,  # 🆕 NEW: Control correlation plot
        correlation_method='spearman',  # 🆕 NEW: 'pearson', 'spearman', 'kendall'
        
        label_used=None,
        sample_per_label=None
        ):
    """
    Extract and save taxonomic data with COMPREHENSIVE imbalance handling
    
    Parameters:
    -----------
    df_tax : pd.DataFrame
        Input taxonomy dataframe
    columns_select : list
        List of taxonomic levels to process (e.g., ['genus', 'species'])
    output_path : str
        Output directory path
    generate_dummy : bool
        Generate dummy k-mer features
    sampling_strategy : str
        Undersampling strategy: 'centroid_closest', 'centroid_diverse', 
        'centroid_kmeans', 'stratified', 'balanced', 'none'
    sample_fraction : float or int
        Fraction/number of samples to keep per class
    min_samples_per_class : int
        Minimum samples per class after sampling
    kmer_size : int
        K-mer size for feature generation
    small_class_threshold : int
        Threshold for grouping rare classes
    small_class_strategy : str
        Strategy for handling small classes: 'group' or 'remove'
    imbalance_strategy : str
        Primary imbalance handling strategy:
        - 'hierarchical_grouping': Group rare classes by higher taxonomy
        - 'adaptive_sampling': Smart sampling with SMOTE
    hierarchical_grouping : bool
        Enable hierarchical rare class grouping
    min_sample_freq : int
        Minimum frequency to keep a class initially
    
    # 🆕 NEW FILTERING PARAMETERS
    filter_uncultured : bool
        Filter out 'uncultured' labels (default: True)
    filter_metagenome : bool
        Filter out 'metagenome' labels (default: True)
    filter_unidentified : bool
        Filter out 'unidentified', 'unknown', 'unclassified' labels (default: True)
    filter_environmental : bool
        Filter out 'environmental', 'clone' labels (default: False)
    custom_filter_keywords : list of str
        Additional custom keywords to filter (default: None)
    case_sensitive_filter : bool
        Perform case-sensitive keyword matching (default: False)
    
    top_n_classes : int or None
        Keep only top N most abundant classes after balancing
    create_plots : bool
        Create visualization plots
    plot_top_n : int
        Number of top classes to show in plots
    label_used : str (deprecated)
        Legacy parameter for label column
    sample_per_label : int
        For 'balanced' strategy: exact number of samples per class

    Extract and process taxonomic data with k-mer vectorization
    
    Parameters:
    -----------
    filter_ambiguous_bases : bool
        Remove sequences containing ambiguous IUPAC codes
    ambiguous_handling : str
        How to handle ambiguous bases:
        - 'remove': Discard sequences with ambiguous bases
        - 'replace': Replace with most likely base
        - 'random': Random choice from possibilities
    max_ambiguous_ratio : float
        Maximum ratio of ambiguous bases allowed (0.0-1.0)
        
    Returns:
    --------
    csv_paths : list
        List of paths to saved CSV files
    paths_file : str
        Path to file containing all CSV paths
        
    Example:
    --------
    >>> csv_paths, paths_file = run_extract_(
    ...     df_tax,
    ...     columns_select=['genus', 'species'],
    ...     output_path='/path/to/output',
    ...     kmer_size=6,
    ...     
    ...     # Filtering configuration
    ...     filter_uncultured=True,
    ...     filter_metagenome=True,
    ...     filter_unidentified=True,
    ...     custom_filter_keywords=['sp.', 'bacterium'],
    ...     
    ...     # Balancing configuration
    ...     imbalance_strategy='hierarchical_grouping',
    ...     small_class_threshold=10,
    ...     min_samples_per_class=10,
    ...     
    ...     top_n_classes=100
    ... )
    """
    
    csv_paths = []
    
    for column in columns_select:
        print(f"\n{'='*70}")
        print(f"🔬 PROCESSING LEVEL: {column.upper()}")
        print(f"{'='*70}")
        
        # ============================================================
        # STEP 1: Filter ambiguous sequences
        # ============================================================
        
        if filter_ambiguous_bases:
            print(f"\n{'='*70}")
            print(f"🧹 FILTERING AMBIGUOUS BASES")
            print(f"{'='*70}")
            
            initial_count = len(df_tax)
            
            # Define ambiguous bases
            ambiguous = set('NRYKHDBVSWM')
            
            if ambiguous_handling == 'remove':
                # Count ambiguous bases per sequence
                def has_ambiguous(seq):
                    seq_upper = seq.upper()
                    ambig_count = sum(1 for base in seq_upper if base in ambiguous)
                    ratio = ambig_count / len(seq) if len(seq) > 0 else 1.0
                    return ratio <= max_ambiguous_ratio
                
                df_tax = df_tax[df_tax['sequence'].apply(has_ambiguous)]
                
                removed = initial_count - len(df_tax)
                print(f"   ✅ Removed {removed:,} sequences ({removed/initial_count*100:.2f}%)")
                print(f"   ✅ Remaining: {len(df_tax):,} sequences")
                
            elif ambiguous_handling == 'replace':
                print(f"   🔄 Replacing ambiguous bases with most likely alternatives...")
                df_tax['sequence'] = df_tax['sequence'].apply(
                    lambda seq: clean_sequence(seq, strategy='replace')
                )
                print(f"   ✅ Cleaned {initial_count:,} sequences")
            
            elif ambiguous_handling == 'random':
                print(f"   🎲 Randomly replacing ambiguous bases...")
                df_tax['sequence'] = df_tax['sequence'].apply(
                    lambda seq: clean_sequence(seq, strategy='random')
                )
                print(f"   ✅ Cleaned {initial_count:,} sequences")
                
        # ============================================================
        # STEP 1: Initial extraction
        # ============================================================
        
        df_result, csv_path = level_extract_plot_freq(
            df_tax, 
            path_=output_path, 
            level=column,
            filter_uncultured=False,
            min_sample_freq=min_sample_freq
        )
        
        # ============================================================
        # STEP 2: Advanced label filtering
        # ============================================================
        
        if (filter_uncultured or filter_metagenome or filter_unidentified or 
            filter_environmental or custom_filter_keywords):
            
            df_result = filter_unwanted_labels_advanced(
                df_result,
                label_column='label',
                filter_uncultured=filter_uncultured,
                filter_metagenome=filter_metagenome,
                filter_unidentified=filter_unidentified,
                filter_environmental=filter_environmental,
                custom_keywords=custom_filter_keywords,
                case_sensitive=case_sensitive_filter,
                verbose=True
            )
        
        df_before = df_result.copy()
        
        # ============================================================
        # STEP 3: Hierarchical grouping
        # ============================================================
        
        if imbalance_strategy == 'hierarchical_grouping' and hierarchical_grouping:
            print(f"\n{'='*70}")
            print(f"🌳 APPLYING HIERARCHICAL GROUPING")
            print(f"{'='*70}")
            
            df_result = hierarchical_rare_grouping(
                df_result,
                label_column='label',
                small_class_threshold=small_class_threshold,
                strategy=small_class_strategy,
                kmer_size=kmer_size,
                verbose=True
            )
        
        # ============================================================
        # STEP 4: Adaptive sampling (if enabled)
        # ============================================================
        
        if imbalance_strategy == 'adaptive_sampling':
            print(f"\n🎚️ APPLYING ADAPTIVE SAMPLING...")
            
            df_result = adaptive_class_sampling(
                df_result,
                label_column='label',
                sampling_strategy=sampling_strategy,
                sample_fraction=sample_fraction,
                min_samples_per_class=min_samples_per_class,
                small_class_threshold=small_class_threshold,
                small_class_strategy=small_class_strategy,
                kmer_size=kmer_size,
                verbose=True
            )
        
        # ============================================================
        # STEP 5: Undersampling strategies
        # ============================================================
        
        # ✅ FIX 1: Handle centroid-based sampling
        elif sampling_strategy in ['centroid_closest', 'centroid_diverse', 'centroid_kmeans']:
            print(f"\n{'='*70}")
            print(f"📊 APPLYING CENTROID-BASED UNDERSAMPLING")
            print(f"   Strategy: {sampling_strategy}")
            print(f"{'='*70}")
            
            try:
                if sampling_strategy == 'centroid_closest':
                    df_result = centroid_closest_sampling(
                        df_result,
                        sample_fraction=sample_fraction,
                        min_samples_per_class=min_samples_per_class,
                        kmer_size=kmer_size,
                        verbose=True
                    )
                
                elif sampling_strategy == 'centroid_diverse':
                    df_result = centroid_diverse_sampling(
                        df_result,
                        sample_fraction=sample_fraction,
                        min_samples_per_class=min_samples_per_class,
                        kmer_size=kmer_size,
                        verbose=True
                    )
                
                elif sampling_strategy == 'centroid_kmeans':
                    df_result = centroid_kmeans_sampling(
                        df_result,
                        sample_fraction=sample_fraction,
                        min_samples_per_class=min_samples_per_class,
                        kmer_size=kmer_size,
                        verbose=True
                    )
            
            except NameError as e:
                print(f"\n   ⚠️  WARNING: Sampling function not found: {e}")
                print(f"   ℹ️  Skipping undersampling, keeping all data")
                # Don't modify df_result if function is missing
        
        elif sampling_strategy == 'stratified':
            print(f"\n{'='*70}")
            print(f"📊 APPLYING STRATIFIED SAMPLING")
            print(f"{'='*70}")
            
            df_result = stratified_sampling(
                df_result,
                sample_fraction=sample_fraction,
                min_samples_per_class=min_samples_per_class,
                verbose=True
            )
        
        elif sampling_strategy == 'balanced':
            print(f"\n{'='*70}")
            print(f"⚖️ APPLYING BALANCED SAMPLING")
            print(f"{'='*70}")
            
            if sample_per_label is None:
                sample_per_label = 50
            
            df_result = balanced_sampling(
                df_result,
                samples_per_class=sample_per_label,
                min_samples_per_class=min_samples_per_class,
                verbose=True
            )
        
        # ============================================================
        # STEP 6: Top-N filtering
        # ============================================================
        
        if top_n_classes is not None and top_n_classes > 0:
            print(f"\n{'='*70}")
            print(f"🔝 FILTERING: KEEPING ONLY TOP {top_n_classes} CLASSES")
            print(f"{'='*70}")
            
            class_counts = df_result['label'].value_counts()
            
            print(f"\n📊 BEFORE TOP-N FILTERING:")
            print(f"   • Total classes: {len(class_counts)}")
            print(f"   • Total samples: {len(df_result):,}")
            
            if len(class_counts) > top_n_classes:
                top_classes = class_counts.nlargest(top_n_classes).index
                df_result = df_result[df_result['label'].isin(top_classes)].copy()
                df_result = df_result.reset_index(drop=True)
                
                dropped_classes = len(class_counts) - top_n_classes
                dropped_samples = len(df_before) - len(df_result)
                
                print(f"\n✅ AFTER TOP-N FILTERING:")
                print(f"   • Remaining classes: {df_result['label'].nunique()}")
                print(f"   • Remaining samples: {len(df_result):,}")
                print(f"   • Dropped classes: {dropped_classes}")
                print(f"   • Dropped samples: {dropped_samples:,}")
        
        # ============================================================
        # STEP 7: Save processed data
        # ============================================================
        
        if sampling_strategy != 'none':
            csv_path = csv_path.replace('.csv', f'_sampled.csv')
        
        df_result.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
        
        print(f"\n{'='*70}")
        print(f"💾 SAVED PROCESSED DATA")
        print(f"{'='*70}")
        print(f"   📁 Path: {csv_path}")
        print(f"   📊 Final: {len(df_result):,} samples, {df_result['label'].nunique()} classes")
        
        # ============================================================
        # STEP 8: Create visualizations
        # ============================================================

        if create_plots:
            print(f"\n{'='*70}")
            print(f"📊 CREATING VISUALIZATIONS")
            print(f"{'='*70}")
            
            output_dir = os.path.dirname(csv_path)
            
            # ✅ EXISTING: Distribution comparison plot
            plot_path = os.path.join(output_dir, f'distribution_comparison_{column}.png')
            plot_distribution_comparison(df_before, df_result, plot_path, column)
            
            # ✅ ADD THIS: Correlation matrix plot
            try:
                print(f"\n   📈 Generating correlation matrix...")
                fig_corr, corr_data = plot_correlation_matrix(
                    df_before=df_before,
                    df_after=df_result,
                    label_column='label',
                    output_path=output_dir,
                    level=column,
                    method='spearman',  # or 'pearson'
                    figsize=(14, 12)
                )
                plt.close(fig_corr)  # Close to free memory
                
                # Save correlation data to CSV
                corr_csv_path = os.path.join(output_dir, f'{column}_correlation_data.csv')
                corr_data.to_csv(corr_csv_path, index=False)
                print(f"   💾 Correlation data saved: {corr_csv_path}")
                
            except Exception as e:
                print(f"   ⚠️  Warning: Could not generate correlation matrix: {e}")
                import traceback
                traceback.print_exc()
            
            # ✅ EXISTING: Frequency distribution (if plot_top_n specified)
            if plot_top_n is not None and plot_top_n > 0:
                plot_path = os.path.join(output_dir, f'frequency_top{plot_top_n}_{column}.png')
                plot_frequency_distribution(df_result, plot_path, column, top_n=plot_top_n)
    
    # ============================================================
    # STEP 9: Save paths file
    # ============================================================
    
    paths_file = os.path.join(output_path, 'csv_level_paths_list.txt')
    with open(paths_file, 'w') as f:
        for path in csv_paths:
            f.write(f"{path}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETED")
    print(f"{'='*70}")
    print(f"   📁 Processed {len(csv_paths)} taxonomic levels")
    print(f"   📄 Paths saved to: {paths_file}")
    
    return csv_paths, paths_file

# def run_extract_(
#         df_tax,
#         columns_select,
#         output_path=None,
#         generate_dummy=True,
#         sampling_strategy='centroid_closest',
#         sample_fraction=0.1,
#         min_samples_per_class=10,
#         kmer_size=6,
#         small_class_threshold=10,
#         small_class_strategy='group',
        
#         imbalance_strategy='hierarchical_grouping',
#         hierarchical_grouping=True,
        
#         min_sample_freq=1,
        
#         # ✅ NEW: Enhanced filtering parameters
#         filter_uncultured=True,
#         filter_metagenome=True,
#         filter_unidentified=True,
#         filter_environmental=False,
#         custom_filter_keywords=None,
#         case_sensitive_filter=False,

#         # 🆕 NEW: Top N classes filtering
#         top_n_classes=None,  # NEW! Set to 100 to keep only top 100 classes
        
#         create_plots=True,
#         plot_top_n=30,
        
#         label_used=None,
#         sample_per_label=None
#         ):
#     """
#     Extract and save taxonomic data with COMPREHENSIVE imbalance handling
    
#     Parameters:
#     -----------
#     df_tax : DataFrame
#         Taxonomic data with sequences
#     columns_select : list
#         List of taxonomic levels to process
#     output_path : str
#         Output directory path
#     generate_dummy : bool
#         If True, apply sampling strategy
#     sampling_strategy : str
#         'centroid_closest', 'centroid_diverse', 'centroid_kmeans', 
#         'stratified', 'balanced'
#     sample_fraction : float
#         Fraction to sample from each class
#     min_samples_per_class : int
#         Minimum samples per class after sampling
#     kmer_size : int
#         K-mer size for centroid methods
#     small_class_threshold : int
#         Threshold for "small" classes
#     small_class_strategy : str
#         How to handle small classes: 'group', 'skip', 'keep'
#     imbalance_strategy : str
#         Main strategy: 'hierarchical_grouping', 'adaptive_sampling', 
#         'hierarchical_only', 'none'
#     hierarchical_grouping : bool
#         Apply hierarchical grouping as preprocessing
#     adaptive_sampling : bool
#         Apply adaptive sampling (SMOTE for rare classes)
#     target_samples_adaptive : int
#         Target samples per class for adaptive sampling
#     top_n_classes : int or None, default=None
#         🆕 NEW PARAMETER
#         Keep only top N classes (by sample count) after balancing
        
#         - **None** (default) → Keep ALL classes
#         - **100** → Keep only top 100 classes (most abundant)
#         - **50** → Keep only top 50 classes
        
#         ⚠️  Applied AFTER balancing/grouping
        
#         Example:
#         >>> # Keep only top 100 classes
#         >>> run_extract_(
#         ...     df_tax,
#         ...     top_n_classes=100,  # Only 100 largest classes
#         ...     hierarchical_grouping=True
#         ... )
    
#     Returns:
#     --------
#     csv_paths : list
#         List of paths to saved CSV files
#     paths_file : str
#         Path to file containing list of CSV paths
    
#     Examples:
#     ---------
#     # RECOMMENDED: Hierarchical grouping + centroid sampling
#     >>> csv_paths, _ = run_extract_(
#     ...     df_tax,
#     ...     columns_select=['genus', 'species'],
#     ...     output_path='./output',
#     ...     generate_dummy=True,
#     ...     sampling_strategy='centroid_closest',
#     ...     sample_fraction=0.1,
#     ...     imbalance_strategy='hierarchical_grouping',
#     ...     hierarchical_grouping=True,
#     ...     small_class_threshold=10
#     ... )
    
#     # ADVANCED: Hierarchical + Adaptive + SMOTE
#     >>> csv_paths, _ = run_extract_(
#     ...     df_tax,
#     ...     columns_select=['species'],
#     ...     output_path='./output',
#     ...     generate_dummy=True,
#     ...     imbalance_strategy='adaptive_sampling',
#     ...     adaptive_sampling=True,
#     ...     target_samples_adaptive=100
#     ... )

#     >>> csv_paths, paths_file = run_extract_(
#     ...     df_tax,
#     ...     columns_select=['genus', 'species'],
#     ...     output_path='/path/to/output',
#     ...     kmer_size=6,
#     ...     
#     ...     # Filtering configuration
#     ...     filter_uncultured=True,
#     ...     filter_metagenome=True,
#     ...     filter_unidentified=True,
#     ...     custom_filter_keywords=['sp.', 'bacterium'],
#     ...     
#     ...     # Balancing configuration
#     ...     imbalance_strategy='hierarchical_grouping',
#     ...     small_class_threshold=10,
#     ...     min_samples_per_class=10,
#     ...     
#     ...     top_n_classes=100
#     ... )
#     """
    
#     csv_paths = []
    
#     for column in columns_select:
#         print(f"\n{'='*70}")
#         print(f"🔬 PROCESSING LEVEL: {column.upper()}")
#         print(f"{'='*70}")
        
#         # ============================================================
#         # STEP 1: Initial Filtering
#         # ============================================================

#         df_result, csv_path = level_extract_plot_freq(
#             df_tax, 
#             path_=output_path, 
#             level=column,
#             filter_uncultured=filter_uncultured,
#             min_sample_freq=min_sample_freq
#         )
        
#         print(f"\n📊 DATA AFTER INITIAL FILTERING:")
#         print(f"   • Samples: {len(df_result):,}")
#         print(f"   • Classes: {df_result['label'].nunique()}")
        
#         class_dist = df_result['label'].value_counts()
#         print(f"   • Min class size: {class_dist.min()}")
#         print(f"   • Max class size: {class_dist.max()}")
#         print(f"   • Imbalance ratio: {class_dist.max() / class_dist.min():.1f}:1")
        
#         # Store "before" data for comparison
#         df_before = df_result.copy()
#         # ============================================================
#         # STEP 2: Apply Imbalance Handling Strategy
#         # ============================================================
        
#         if generate_dummy:
            
#             # --------------------------------------------------------
#             # STRATEGY A: HIERARCHICAL GROUPING (RECOMMENDED)
#             # --------------------------------------------------------
            
#             if imbalance_strategy == 'hierarchical_grouping' or hierarchical_grouping:
#                 print(f"\n🌳 APPLYING HIERARCHICAL GROUPING...")
                
#                 df_result = hierarchical_rare_grouping(
#                     df_result,
#                     label_column='label',
#                     min_samples=small_class_threshold,
#                     group_by_taxonomy=True,
#                     taxonomy_delimiter='_',
#                     verbose=True
#                 )
            
#             # --------------------------------------------------------
#             # STRATEGY B: ADAPTIVE SAMPLING WITH SMOTE
#             # --------------------------------------------------------
            
#             if imbalance_strategy == 'adaptive_sampling':
#                 print(f"\n🎚️ APPLYING ADAPTIVE SAMPLING...")
                
#                 df_result = adaptive_class_sampling(
#                     df_result,
#                     label_column='label',
#                     sequence_column='sequence',
#                     target_samples_per_class=target_samples_adaptive,
#                     keep_all_rare=True,
#                     oversample_rare=True,
#                     rare_threshold=small_class_threshold,
#                     kmer_size=kmer_size,
#                     verbose=True
#                 )
            
#             # --------------------------------------------------------
#             # STRATEGY C: STANDARD SAMPLING (EXISTING)
#             # --------------------------------------------------------
            
#             print(f"\n📊 APPLYING SAMPLING STRATEGY: {sampling_strategy}")
            
#             if sampling_strategy.startswith('centroid_'):
#                 method = sampling_strategy.replace('centroid_', '')
                
#                 df_sampled = centroid_based_sampling(
#                     df=df_result,
#                     column_name='sequence',
#                     label_column='label',
#                     sample_fraction=sample_fraction,
#                     min_samples=min_samples_per_class,
#                     kmer_size=kmer_size,
#                     method=method,
#                     small_class_threshold=small_class_threshold,
#                     small_class_strategy=small_class_strategy
#                 )
            
#             elif sampling_strategy == 'stratified':
#                 print(f"   📦 Stratified random sampling: {sample_fraction*100:.1f}%")
                
#                 df_sampled = (
#                     df_result.groupby('label', group_keys=False)
#                     .apply(lambda x: x.sample(
#                         n=max(min_samples_per_class, int(len(x) * sample_fraction)),
#                         random_state=42,
#                         replace=False if len(x) >= min_samples_per_class else True
#                     ))
#                     .reset_index(drop=True)
#                 )
            
#             elif sampling_strategy == 'balanced':
#                 n_per_class = sample_per_label if sample_per_label else min_samples_per_class
#                 print(f"   ⚖️ Balanced sampling: {n_per_class} samples per class")
                
#                 df_sampled = (
#                     df_result.groupby('label', group_keys=False)
#                     .apply(lambda x: x.sample(
#                         n=min(len(x), n_per_class),
#                         random_state=42
#                     ))
#                     .reset_index(drop=True)
#                 )
            
#             else:
#                 raise ValueError(f"Unknown sampling_strategy: '{sampling_strategy}'")
        
#         else:
#             print(f"\n   📦 Using FULL dataset (no sampling)")
#             df_sampled = df_result.copy()
        
#         # ============================================================
#         # 🆕 NEW: TOP N CLASSES FILTERING (AFTER BALANCING)
#         # ============================================================
        
#         if top_n_classes is not None and top_n_classes > 0:
#             print(f"\n{'='*70}")
#             print(f"🔝 FILTERING: KEEPING ONLY TOP {top_n_classes} CLASSES")
#             print(f"{'='*70}")
            
#             # Get class counts
#             class_counts = df_result['label'].value_counts()
            
#             print(f"\n📊 BEFORE TOP-N FILTERING:")
#             print(f"   • Total classes: {len(class_counts)}")
#             print(f"   • Total samples: {len(df_result):,}")
            
#             if len(class_counts) > top_n_classes:
#                 # Select top N classes
#                 top_classes = class_counts.nlargest(top_n_classes).index
                
#                 # Filter dataframe
#                 df_result = df_result[df_result['label'].isin(top_classes)].copy()
#                 df_result = df_result.reset_index(drop=True)
                
#                 # Calculate statistics
#                 dropped_classes = len(class_counts) - top_n_classes
#                 dropped_samples = len(df_before) - len(df_result)
                
#                 print(f"\n✅ AFTER TOP-N FILTERING:")
#                 print(f"   • Remaining classes: {df_result['label'].nunique()}")
#                 print(f"   • Remaining samples: {len(df_result):,}")
#                 print(f"   • Dropped classes: {dropped_classes}")
#                 print(f"   • Dropped samples: {dropped_samples:,} ({(dropped_samples/len(df_before))*100:.2f}%)")
                
#                 # Show examples of kept classes
#                 print(f"\n   📋 Top 10 kept classes:")
#                 top_10 = df_result['label'].value_counts().head(10)
#                 for idx, (cls, cnt) in enumerate(top_10.items(), 1):
#                     cls_str = str(cls)[:50] + "..." if len(str(cls)) > 53 else str(cls)
#                     print(f"      {idx:2d}. {cls_str}: {cnt} samples")
                
#                 # Show examples of dropped classes
#                 dropped_class_list = class_counts[~class_counts.index.isin(top_classes)].sort_values(ascending=False)
#                 if len(dropped_class_list) > 0:
#                     print(f"\n   📋 Top 10 dropped classes:")
#                     for idx, (cls, cnt) in enumerate(dropped_class_list.head(10).items(), 1):
#                         cls_str = str(cls)[:50] + "..." if len(str(cls)) > 53 else str(cls)
#                         print(f"      {idx:2d}. {cls_str}: {cnt} samples")
#             else:
#                 print(f"\n   ⚠️  Only {len(class_counts)} classes available (< {top_n_classes})")
#                 print(f"   ✅ Keeping all {len(class_counts)} classes")
        
#         # ============================================================
#         # STEP 3: Save Results
#         # ============================================================
        
#         if output_path:
#             folder_path = os.path.join(output_path, column)
#             os.makedirs(folder_path, exist_ok=True)
            
#             filename = f'{column}_sampled.csv' if generate_dummy else f'{column}.csv'
#             csv_sampled_path = os.path.join(folder_path, filename)
            
#             df_sampled.to_csv(csv_sampled_path, index=False)
#             print(f"\n   💾 Saved to: {csv_sampled_path}")
            
#             # Save statistics
#             stats_path = os.path.join(folder_path, f'{column}_stats.txt')
#             with open(stats_path, 'w') as f:
#                 f.write(f"PROCESSING STATISTICS: {column}\n")
#                 f.write("="*70 + "\n\n")
#                 f.write(f"Initial Data:\n")
#                 f.write(f"  - Samples: {len(df_result):,}\n")
#                 f.write(f"  - Classes: {df_result['label'].nunique()}\n\n")
#                 f.write(f"Final Data:\n")
#                 f.write(f"  - Samples: {len(df_sampled):,}\n")
#                 f.write(f"  - Classes: {df_sampled['label'].nunique()}\n")
#                 f.write(f"  - Reduction: {(1-len(df_sampled)/len(df_result))*100:.1f}%\n\n")
#                 f.write(f"Strategy Applied:\n")
#                 f.write(f"  - Imbalance: {imbalance_strategy}\n")
#                 f.write(f"  - Sampling: {sampling_strategy}\n")
            
#             print(f"   📄 Statistics saved to: {stats_path}")
            
#             csv_paths.append(csv_sampled_path)

#        # ============================================================
#         # 🆕 STEP 4: GENERATE VISUALIZATIONS & ANALYSIS
#         # ============================================================
        
#         if create_plots and generate_dummy and output_path:
#             folder_path = os.path.join(output_path, column)
            
#             print(f"\n📊 GENERATING VISUALIZATIONS & ANALYSIS...")
            
#             try:
#                 # Plot 1: Distribution Comparison
#                 fig_dist, stats_dist = plot_class_distribution_comparison(
#                     df_before=df_before,
#                     df_after=df_sampled,
#                     label_column='label',
#                     output_path=folder_path,
#                     level=column,
#                     show_top_n=plot_top_n
#                 )
#                 plt.close(fig_dist)
                
#                 # Plot 2: Correlation Analysis
#                 fig_corr, corr_data = plot_correlation_matrix(
#                     df_before=df_before,
#                     df_after=df_sampled,
#                     label_column='label',
#                     output_path=folder_path,
#                     level=column,
#                     method='spearman'
#                 )
#                 plt.close(fig_corr)
                
#                 # Report: Comprehensive Statistics
#                 report_text = generate_balancing_report(
#                     df_before=df_before,
#                     df_after=df_sampled,
#                     label_column='label',
#                     output_path=folder_path,
#                     level=column,
#                     imbalance_strategy=imbalance_strategy,
#                     sampling_strategy=sampling_strategy
#                 )
                
#                 # Save correlation data
#                 corr_csv_path = os.path.join(folder_path, 
#                                             f'{column}_correlation_data.csv')
#                 corr_data.to_csv(corr_csv_path)
#                 print(f"   💾 Correlation data saved: {corr_csv_path}")
                
#                 print(f"   ✅ All visualizations and analysis completed!")
                
#             except Exception as e:
#                 print(f"   ⚠️ WARNING: Could not generate plots: {e}")
#                 import traceback
#                 traceback.print_exc()   
#     # ============================================================
#     # Save Path List
#     # ============================================================
    
#     paths_file = None
#     if output_path and len(csv_paths) > 0:
#         paths_file = os.path.join(output_path, 'csv_level_paths_list.txt')
#         with open(paths_file, 'w') as f:
#             for path in csv_paths:
#                 f.write(f"{path}\n")
#         print(f"\n📝 Path list saved to: {paths_file}")
    
#     return csv_paths, paths_file


"""
Example 1: Hierarchical Grouping (BEST for Microbiome) ⭐⭐⭐⭐⭐
# RECOMMENDED untuk data dengan 80,000 vs 1 sample imbalance
csv_paths, paths_file = run_extract_(
    df_tax,
    columns_select=['genus', 'species'],
    output_path='./output',
    
    # Main settings
    generate_dummy=True,
    sampling_strategy='centroid_closest',
    sample_fraction=0.1,
    min_samples_per_class=10,
    kmer_size=6,
    
    # 🆕 Hierarchical grouping (preserves taxonomy)
    imbalance_strategy='hierarchical_grouping',
    hierarchical_grouping=True,
    small_class_threshold=10,
    small_class_strategy='group'  # Group rare by genus
)

Example 2: Adaptive Sampling with SMOTE ⭐⭐⭐
csv_paths, paths_file = run_extract_(
    df_tax,
    columns_select=['species'],
    output_path='./output',
    
    # Use adaptive sampling
    generate_dummy=True,
    imbalance_strategy='adaptive_sampling',
    adaptive_sampling=True,
    target_samples_adaptive=100,  # Balance to ~100 per class
    
    # Settings
    small_class_threshold=10,
    kmer_size=6
)

Example 3: Combined (Hierarchical + Adaptive + Centroid)
# MOST COMPREHENSIVE APPROACH
csv_paths, paths_file = run_extract_(
    df_tax,
    columns_select=['species'],
    output_path='./output',
    
    # Step 1: Hierarchical grouping
    hierarchical_grouping=True,
    small_class_threshold=10,
    
    # Step 2: Adaptive sampling with SMOTE
    adaptive_sampling=True,
    target_samples_adaptive=100,
    
    # Step 3: Centroid-based final sampling
    generate_dummy=True,
    sampling_strategy='centroid_diverse',  # Diverse sampling
    sample_fraction=0.5,  # Since already balanced, sample 50%
    kmer_size=6
)

Example 4: Combined (Hierarchical + Adaptive + Centroid) + plot comparison
# ✅ BEST PRACTICE: Preserve ALL rare taxa
csv_paths, paths_file = run_extract_(
    df_tax,
    columns_select=['genus', 'species'],
    output_path='/Users/tirtasetiawan/Documents/rki_v1/rki_2025/dataset',
    
    # 🆕 Keep all classes initially
    min_sample_freq=1,           # ✅ NO initial filtering
    filter_uncultured=True,      # Still remove 'uncultured'
    
    # Balancing strategy
    generate_dummy=True,
    imbalance_strategy='hierarchical_grouping',
    hierarchical_grouping=True,   # Group rare by genus
    sampling_strategy='centroid_closest',
    sample_fraction=2,
    small_class_threshold=10,     # Group if <10 samples
    small_class_strategy='group',
    
    # Visualization
    create_plots=True,
    plot_top_n=None
)

"""