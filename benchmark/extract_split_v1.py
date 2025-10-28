from scipy import sparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import time

def load_taxonomy_paths_from_txt(txt_file):
    """Load taxonomy file paths from txt file"""
    paths_dict = {}

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    print("🔍 Loading taxonomy file paths...")
    for line in tqdm(lines, desc="Reading paths", unit="file"):
        path = line.strip()
        if path and os.path.exists(path):
            # Extract level from filename
            filename = os.path.basename(path)
            level = filename.replace('.csv', '')
            paths_dict[level] = path
            tqdm.write(f"✅ Found {level}: {path}")
        elif path:
            tqdm.write(f"❌ File not found: {path}")
        time.sleep(0.1)  # Small delay for visual effect

    print(f"\n✅ Loaded {len(paths_dict)} taxonomy files")
    return paths_dict


def load_all_taxonomy_data(taxonomy_paths):
    """Load all taxonomy data from paths with progress monitoring"""
    all_level_data = {}

    print("\n📊 Loading taxonomy data...")
    for level, path in tqdm(taxonomy_paths.items(), desc="Loading data", unit="level"):
        try:
            tqdm.write(f"Loading {level} data from: {path}")
            df = pd.read_csv(path)

            # Rename columns to standard format
            if level in df.columns:
                df = df.rename(columns={level: 'label'})

            # Check required columns
            if 'sequence' not in df.columns or 'label' not in df.columns:
                tqdm.write(f"❌ Missing required columns for {level}")
                continue

            # Remove NaN values with progress
            initial_len = len(df)
            df = df.dropna(subset=['sequence', 'label'])

            all_level_data[level] = df
            tqdm.write(f"✅ Loaded {level}: {len(df)} samples ({initial_len - len(df)} removed), {df['label'].nunique()} unique labels")

        except Exception as e:
            tqdm.write(f"❌ Error loading {level}: {str(e)}")

        time.sleep(0.2)  # Small delay for visual effect

    return all_level_data


def vectorize_data(df, level, k=6, save_path=None, test_size=0.2, random_state=42):
    """Vectorize data for specific level and k-mer with progress monitoring and train-test split"""
    print(f"\n🔄 CountVectorizer k-mer {k} for {level}")

    # Vectorization with progress bar
    print("Creating vectorizer...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))

    print("Fitting and transforming sequences...")
    # Split into chunks for progress monitoring
    chunk_size = max(1, len(df) // 10)

    with tqdm(total=len(df), desc=f"Vectorizing {level} k-mer {k}", unit="seq") as pbar:
        X_sparse = vectorizer.fit_transform(df['sequence'])
        pbar.update(len(df))

    y = df['label'].values

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    with tqdm(total=1, desc="Label encoding", unit="step") as pbar:
        y_encoded = label_encoder.fit_transform(y)
        pbar.update(1)

    print("Shape X:", X_sparse.shape)
    print("Shape y:", y_encoded.shape)

    n_classes = len(label_encoder.classes_)
    print(f"Jumlah kelas unik (label): {n_classes}")
    print("Jumlah total data:", len(y_encoded))

    # Train-test split
    print(f"🔀 Train-test split (test_size={test_size})...")
    try:
        # Use stratify only if all classes have at least 2 samples
        unique_labels, label_counts = np.unique(y_encoded, return_counts=True)
        min_label_count = min(label_counts)
        stratify = y_encoded if min_label_count >= 2 else None
        
        if stratify is None:
            print("⚠️  Some classes have only 1 sample, stratification disabled")

        X_train_sparse, X_test_sparse, y_train, y_test = train_test_split(
            X_sparse,           # Features (sparse matrix dari CountVectorizer)
            y_encoded,          # Target labels (encoded)
            test_size=test_size,      # test size
            random_state=random_state,    # Untuk reproducibility
            stratify=stratify  # Mempertahankan distribusi label jika memungkinkan
        )

        print(f"Train label count: {len(np.unique(y_train))}")
        print(f"Test label count: {len(np.unique(y_test))}")
        print(f"Train shape: X{X_train_sparse.shape}, y{y_train.shape}")
        print(f"Test shape: X{X_test_sparse.shape}, y{y_test.shape}")

        # Validation checks
        assert X_train_sparse.shape[0] == len(y_train), "Train X dan y harus sama panjang"
        assert X_test_sparse.shape[0] == len(y_test), "Test X dan y harus sama panjang"
        assert len(y_encoded) == len(y_train) + len(y_test), "Total data harus sama"

    except Exception as e:
        print(f"❌ Error during train-test split: {e}")
        # Fallback: no split, return original data as train
        X_train_sparse, X_test_sparse = X_sparse, None
        y_train, y_test = y_encoded, None

    # Prepare file paths
    file_paths = {}

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Save files with progress monitoring
        files_to_save = [
            ('label_encoder', f'label_encoder_k{k}_{level}.pkl', label_encoder),
            ('vectorizer', f'vectorizer_k{k}_{level}.pkl', vectorizer),
            ('X_sparse', f"X_sparse_k{k}_{level}.npz", X_sparse),
            ('y_encoded', f"y_encoded_k{k}_{level}.npy", y_encoded),
            ('X_train_sparse', f"X_train_sparse_k{k}_{level}.npz", X_train_sparse),
            ('y_train', f"y_train_k{k}_{level}.npy", y_train),
        ]

        # Add test files if split was successful
        if X_test_sparse is not None:
            files_to_save.extend([
                ('X_test_sparse', f"X_test_sparse_k{k}_{level}.npz", X_test_sparse),
                ('y_test', f"y_test_k{k}_{level}.npy", y_test),
            ])

        with tqdm(files_to_save, desc="Saving files", unit="file") as pbar:
            for file_type, filename, data in pbar:
                pbar.set_postfix_str(f"Saving {filename}")

                file_path = os.path.join(save_path, filename)

                if file_type in ['label_encoder', 'vectorizer']:
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f)
                elif 'sparse' in file_type:
                    sparse.save_npz(file_path, data)
                elif file_type in ['y_encoded', 'y_train', 'y_test']:
                    np.save(file_path, data)

                file_paths[file_type] = file_path
                tqdm.write(f"✅ Saved: {file_path}")
                time.sleep(0.1)

        # Save CSV files for y_train and y_test (optional)
        if X_test_sparse is not None:
            train_csv_path = os.path.join(save_path, f'y_train_k{k}_{level}.csv')
            test_csv_path = os.path.join(save_path, f'y_test_k{k}_{level}.csv')
            
            pd.DataFrame({'y_train': y_train}).to_csv(train_csv_path, index=False)
            pd.DataFrame({'y_test': y_test}).to_csv(test_csv_path, index=False)
            
            file_paths['y_train_csv'] = train_csv_path
            file_paths['y_test_csv'] = test_csv_path
            
            print(f"✅ Saved train/test CSV: {train_csv_path}, {test_csv_path}")

        return X_sparse, y_encoded, label_encoder, file_paths, X_train_sparse, X_test_sparse, y_train, y_test
    else:
        return X_sparse, y_encoded, label_encoder, {}, X_train_sparse, X_test_sparse, y_train, y_test


def process_all_vectorization_from_txt(taxonomy_paths_txt, k_values=[6, 8, 10],
                                      base_save_path=None, test_size=0.2, random_state=42):
    """Process vectorization with comprehensive progress monitoring and train-test split"""

    # Load taxonomy paths from txt file
    taxonomy_paths = load_taxonomy_paths_from_txt(taxonomy_paths_txt)

    # Load all taxonomy data
    all_level_data = load_all_taxonomy_data(taxonomy_paths)

    # Get available levels from loaded data
    taxonomy_levels = list(all_level_data.keys())

    # Calculate total operations for main progress bar
    total_operations = len(taxonomy_levels) * len(k_values)

    # Dictionary to store all file paths
    vectorization_paths = {}
    processing_summary = []
    all_output_paths = []

    print("\n" + "="*80)
    print("🚀 STARTING VECTORIZATION PROCESS WITH TRAIN-TEST SPLIT")
    print("="*80)
    print(f"📊 Total operations: {total_operations}")
    print(f"📁 Levels: {taxonomy_levels}")
    print(f"🔢 K-mers: {k_values}")
    print(f"🔀 Test size: {test_size}")
    print(f"🎲 Random state: {random_state}")
    print("="*80)

    # Main progress bar for overall process
    main_pbar = tqdm(total=total_operations,
                    desc="Overall Progress",
                    unit="operation",
                    position=0,
                    leave=True)

    for level_idx, level in enumerate(taxonomy_levels):
        print(f"\n{'='*60}")
        print(f"🧬 PROCESSING LEVEL: {level.upper()} ({level_idx+1}/{len(taxonomy_levels)})")
        print(f"{'='*60}")

        # Get data for this level
        df = all_level_data[level].copy()
        print(f"✅ Processing {level}: {len(df)} samples, {df['label'].nunique()} unique labels")

        # Initialize level paths
        vectorization_paths[level] = {}

        # Progress bar for k-mers within this level
        k_pbar = tqdm(k_values,
                     desc=f"K-mers for {level}",
                     unit="k-mer",
                     position=1,
                     leave=False)

        for k in k_pbar:
            k_pbar.set_postfix_str(f"Processing k-mer {k}")

            try:
                # Create save path for this level and k-mer
                if base_save_path:
                    save_path = os.path.join(base_save_path, 'vectorization', level, f'k{k}')
                else:
                    save_path = None

                # Update main progress bar description
                main_pbar.set_description(f"Processing {level} k-mer {k}")

                # Vectorize data with train-test split
                result = vectorize_data(
                    df, level, k, save_path, test_size, random_state
                )
                
                X_sparse, y_encoded, label_encoder, file_paths, X_train_sparse, X_test_sparse, y_train, y_test = result

                # Store paths
                vectorization_paths[level][f'k{k}'] = file_paths

                # Add file paths to the all_output_paths list
                for file_type, path in file_paths.items():
                    all_output_paths.append(path)

                # Add to summary
                summary_data = {
                    'level': level,
                    'k_mer': k,
                    'n_samples': X_sparse.shape[0],
                    'n_features': X_sparse.shape[1],
                    'n_classes': len(label_encoder.classes_),
                    'n_train_samples': X_train_sparse.shape[0] if X_train_sparse is not None else 0,
                    'n_test_samples': X_test_sparse.shape[0] if X_test_sparse is not None else 0,
                    'test_size': test_size,
                    'train_test_split': X_test_sparse is not None,
                    'status': 'success'
                }
                processing_summary.append(summary_data)

                tqdm.write(f"✅ Successfully processed {level} k-mer {k}")
                if X_test_sparse is not None:
                    tqdm.write(f"   📊 Split: {X_train_sparse.shape[0]} train, {X_test_sparse.shape[0]} test")

            except Exception as e:
                tqdm.write(f"❌ Error processing {level} k-mer {k}: {str(e)}")
                processing_summary.append({
                    'level': level,
                    'k_mer': k,
                    'status': 'failed',
                    'error': str(e)
                })

            # Update main progress bar
            main_pbar.update(1)
            time.sleep(0.1)

        k_pbar.close()

    main_pbar.close()

    # Save configuration and summary with progress
    if base_save_path:
        print("\n💾 Saving configuration files...")
        save_configuration(vectorization_paths, processing_summary, all_output_paths, base_save_path)

    print("\n" + "="*80)
    print("🎉 VECTORIZATION PROCESS WITH TRAIN-TEST SPLIT COMPLETED")
    print("="*80)

    # Print summary
    print_processing_summary(processing_summary)

    return vectorization_paths, processing_summary, all_output_paths


def save_configuration(vectorization_paths, processing_summary, all_output_paths, base_save_path):
    """Save vectorization configuration and summary with progress monitoring"""
    import json

    config_dir = os.path.join(base_save_path, 'vectorization_config')
    os.makedirs(config_dir, exist_ok=True)

    # Files to save
    save_tasks = [
        ('config', 'vectorization_config.json'),
        ('summary_csv', 'processing_summary.csv'),
        ('all_paths', 'all_vectorization_output_paths.txt'),
        ('organized_paths', 'organized_vectorization_paths.txt'),
        ('by_type_paths', 'paths_by_file_type.txt')
    ]

    with tqdm(save_tasks, desc="Saving config files", unit="file") as pbar:
        for task_type, filename in pbar:
            pbar.set_postfix_str(f"Saving {filename}")
            file_path = os.path.join(config_dir, filename)

            if task_type == 'config':
                config_data = {
                    'timestamp': datetime.now().isoformat(),
                    'vectorization_paths': vectorization_paths,
                    'summary': processing_summary
                }
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

            elif task_type == 'summary_csv':
                summary_df = pd.DataFrame(processing_summary)
                summary_df.to_csv(file_path, index=False)

            elif task_type == 'all_paths':
                with open(file_path, 'w') as f:
                    for path in sorted(all_output_paths):
                        f.write(f"{path}\n")

            elif task_type == 'organized_paths':
                with open(file_path, 'w') as f:
                    f.write("VECTORIZATION FILE PATHS (ORGANIZED)\n")
                    f.write("="*50 + "\n\n")

                    for level, k_data in vectorization_paths.items():
                        f.write(f"LEVEL: {level.upper()}\n")
                        f.write("-" * 30 + "\n")

                        for k_mer, paths in k_data.items():
                            f.write(f"  {k_mer}:\n")
                            for file_type, path in paths.items():
                                f.write(f"    {file_type}: {path}\n")
                            f.write("\n")
                        f.write("\n")

            elif task_type == 'by_type_paths':
                with open(file_path, 'w') as f:
                    f.write("VECTORIZATION PATHS BY FILE TYPE\n")
                    f.write("="*40 + "\n\n")

                    # Group by file type
                    by_type = {}
                    for level, k_data in vectorization_paths.items():
                        for k_mer, paths in k_data.items():
                            for file_type, path in paths.items():
                                if file_type not in by_type:
                                    by_type[file_type] = []
                                by_type[file_type].append(path)

                    for file_type, paths in by_type.items():
                        f.write(f"{file_type.upper()} FILES:\n")
                        f.write("-" * 20 + "\n")
                        for path in sorted(paths):
                            f.write(f"{path}\n")
                        f.write("\n")

            tqdm.write(f"✅ Saved: {file_path}")
            time.sleep(0.1)


def print_processing_summary(processing_summary):
    """Print processing summary with enhanced formatting including train-test split info"""
    successful = [s for s in processing_summary if s['status'] == 'success']
    failed = [s for s in processing_summary if s['status'] == 'failed']

    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        print(f"\n✅ SUCCESSFUL PROCESSES:")
        for s in tqdm(successful, desc="Showing successful", leave=False):
            train_info = f", train: {s.get('n_train_samples', 'N/A')}, test: {s.get('n_test_samples', 'N/A')}" if s.get('train_test_split', False) else ""
            print(f"  - {s['level']} k-mer {s['k_mer']}: {s['n_samples']} samples, {s['n_features']} features, {s['n_classes']} classes{train_info}")

    if failed:
        print(f"\n❌ FAILED PROCESSES:")
        for s in failed:
            print(f"  - {s['level']} k-mer {s['k_mer']}: {s['error']}")


def load_vectorized_data_from_path(X_sparse_path, y_encoded_path, label_encoder_path, vectorizer_path):
    """Load specific vectorized data from individual file paths with progress"""

    files_to_load = [
        ("X_sparse", X_sparse_path),
        ("y_encoded", y_encoded_path),
        ("label_encoder", label_encoder_path),
        ("vectorizer", vectorizer_path)
    ]

    results = {}

    with tqdm(files_to_load, desc="Loading vectorized data", unit="file") as pbar:
        for file_type, path in pbar:
            pbar.set_postfix_str(f"Loading {file_type}")

            if file_type == "X_sparse":
                results[file_type] = sparse.load_npz(path)
            elif file_type == "y_encoded":
                results[file_type] = np.load(path)
            elif file_type in ["label_encoder", "vectorizer"]:
                with open(path, 'rb') as f:
                    results[file_type] = pickle.load(f)

            time.sleep(0.1)

    print(f"✅ Loaded data: X{results['X_sparse'].shape}, y{results['y_encoded'].shape}")

    return results['X_sparse'], results['y_encoded'], results['label_encoder'], results['vectorizer']


def load_train_test_data_from_path(train_X_path, test_X_path, train_y_path, test_y_path, label_encoder_path, vectorizer_path):
    """Load train-test split data from individual file paths with progress"""

    files_to_load = [
        ("X_train_sparse", train_X_path),
        ("X_test_sparse", test_X_path),
        ("y_train", train_y_path),
        ("y_test", test_y_path),
        ("label_encoder", label_encoder_path),
        ("vectorizer", vectorizer_path)
    ]

    results = {}

    with tqdm(files_to_load, desc="Loading train-test data", unit="file") as pbar:
        for file_type, path in pbar:
            pbar.set_postfix_str(f"Loading {file_type}")

            if "sparse" in file_type:
                results[file_type] = sparse.load_npz(path)
            elif file_type in ["y_train", "y_test"]:
                results[file_type] = np.load(path)
            elif file_type in ["label_encoder", "vectorizer"]:
                with open(path, 'rb') as f:
                    results[file_type] = pickle.load(f)

            time.sleep(0.1)

    print(f"✅ Loaded train-test data:")
    print(f"   Train: X{results['X_train_sparse'].shape}, y{results['y_train'].shape}")
    print(f"   Test: X{results['X_test_sparse'].shape}, y{results['y_test'].shape}")

    return (results['X_train_sparse'], results['X_test_sparse'], 
            results['y_train'], results['y_test'], 
            results['label_encoder'], results['vectorizer'])


# Main usage function with enhanced monitoring
def main_vectorization_process(taxonomy_paths_txt, output_base_path, k_values=[6, 8, 10], test_size=0.2, random_state=42):
    """Main function to run the complete vectorization process with full monitoring and train-test split"""

    print("🚀 STARTING VECTORIZATION PIPELINE WITH TRAIN-TEST SPLIT")
    print("="*60)
    print(f"📁 Input taxonomy paths file: {taxonomy_paths_txt}")
    print(f"📁 Output base path: {output_base_path}")
    print(f"🔢 K-mer values: {k_values}")
    print(f"🔀 Test size: {test_size}")
    print(f"🎲 Random state: {random_state}")
    print("="*60)

    start_time = time.time()

    # Process all vectorizations with train-test split
    vectorization_paths, summary, all_output_paths = process_all_vectorization_from_txt(
        taxonomy_paths_txt=taxonomy_paths_txt,
        k_values=k_values,
        base_save_path=output_base_path,
        test_size=test_size,
        random_state=random_state
    )

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n🎉 VECTORIZATION PIPELINE WITH TRAIN-TEST SPLIT COMPLETED!")
    print(f"⏱️  Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"📊 Total files created: {len(all_output_paths)}")

    return vectorization_paths, summary, all_output_paths

from tqdm import tqdm
import time

def batch_load_with_monitoring(output_paths_txt, levels, k_mers):
    """Load multiple vectorizations with progress monitoring"""

    total_combinations = len(levels) * len(k_mers)

    loaded_data = {}

    with tqdm(total=total_combinations, desc="Loading vectorizations", unit="combination") as main_pbar:
        for level in levels:
            loaded_data[level] = {}

            for k in k_mers:
                main_pbar.set_description(f"Loading {level} k-mer {k}")

                try:
                    X, y, label_enc, vectorizer = quick_load_from_txt(output_paths_txt, level, k)
                    loaded_data[level][f'k{k}'] = {
                        'X': X,
                        'y': y,
                        'label_encoder': label_enc,
                        'vectorizer': vectorizer
                    }
                    tqdm.write(f"✅ Loaded {level} k-mer {k}")
                except Exception as e:
                    tqdm.write(f"❌ Failed to load {level} k-mer {k}: {str(e)}")

                main_pbar.update(1)
                time.sleep(0.1)

    return loaded_data

def quick_load_from_txt(output_paths_txt, level, k_mer, file_type='all'):
    """Quick load vectorized data using the output paths txt file with progress"""

    # Read all paths
    with open(output_paths_txt, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]

    # Filter paths for specific level and k-mer
    relevant_paths = {}

    for path in all_paths:
        filename = os.path.basename(path)
        if f'k{k_mer}_{level}' in filename:
            if 'label_encoder' in filename:
                relevant_paths['label_encoder'] = path
            elif 'vectorizer' in filename:
                relevant_paths['vectorizer'] = path
            elif 'X_sparse' in filename:
                relevant_paths['X_sparse'] = path
            elif 'y_encoded' in filename:
                relevant_paths['y_encoded'] = path

    if file_type == 'all':
        # Load all data with progress
        return load_vectorized_data_from_path(
            relevant_paths['X_sparse'],
            relevant_paths['y_encoded'],
            relevant_paths['label_encoder'],
            relevant_paths['vectorizer']
        )
    else:
        # Return specific path
        return relevant_paths.get(file_type)
