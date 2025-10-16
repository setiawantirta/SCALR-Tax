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
    folder_path = None  # Initialize folder_path first

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

def run_extract_ (
        df_tax,
        columns_select,
        output_path=None,
        generate_dummy=True,
        label_used = 10,
        sample_per_label = 10
        ):
    if generate_dummy:
        """
        Ambil 10 label terbanyak dari tiap level, lalu ambil sample 30 baris per label.
        Simpan hasil dummy CSV, dan simpan daftar path file ke file txt.
        """
        csv_dummy_paths = []

        for column in columns_select:
            # Gunakan proses filter awal
            df_result, csv_path = level_extract_plot_freq(df_tax, path_=output_path, level=column)

            # Hitung frekuensi label
            label_counts = df_result['label'].value_counts()

            if label_used is not None:
                # Ambil 10 label dengan frekuensi tertinggi
                top_labels = label_counts.head(label_used).index

                # Filter hanya 10 label ini
                df_top = df_result[df_result['label'].isin(top_labels)].copy()
            else:
                df_top = df_result.copy()
                
            # Ambil sample 10 baris/label
            df_dummy = (
                df_top.groupby('label', group_keys=False)
                      .apply(lambda x: x.sample(min(len(x), sample_per_label), random_state=42))
                      .reset_index(drop=True)
            )

            # Simpan CSV dummy jika ada output_path
            csv_dummy_path = None
            if output_path:
                folder_path = os.path.join(output_path, f"{column}")
                os.makedirs(folder_path, exist_ok=True)
                csv_dummy_path = os.path.join(folder_path, f'{column}.csv')
                df_dummy.to_csv(csv_dummy_path, index=False)
                print(f"CSV dummy saved to: {csv_dummy_path}")

            if csv_dummy_path:
                csv_dummy_paths.append(csv_dummy_path)

            print(f"[{column}] Jumlah label dummy: {df_dummy['label'].nunique()}")
            print(f"[{column}] Jumlah baris dummy: {len(df_dummy)}")
            print("========================================================")

        # Simpan daftar path ke file txt jika ada output_path
        paths_file = None
        if output_path and len(csv_dummy_paths) > 0:
            paths_file = os.path.join(output_path, 'csv_level_paths_list.txt')
            with open(paths_file, 'w') as f:
                for path in csv_dummy_paths:
                    f.write(f"{path}\n")
            print(f"Daftar path dummy CSV disimpan di: {paths_file}")

        return csv_dummy_paths, paths_file

    else:
        # Usage: Simpan semua path ke dalam list
        csv_level_file_path = []

        for column in columns_select:
            df_result, csv_path = level_extract_plot_freq(df_tax, path_=output_path, level=column)
            if csv_path:  # Pastikan path tidak None
                csv_level_file_path.append(csv_path)

        # Print semua path
        print("\n=== ALL CSV FILE PATHS ===")
        for i, path in enumerate(csv_level_file_path, 1):
            print(f"{i}. {path}")

        # Simpan list ke file text - FIXED: gunakan output_path bukan folders['dataset']
        paths_file = None
        if output_path:
            paths_file = os.path.join(output_path, 'csv_level_paths_list.txt')
            with open(paths_file, 'w') as f:
                for path in csv_level_file_path:
                    f.write(f"{path}\n")
            print(f"\nPaths level list saved to: {paths_file}")
        
        return csv_level_file_path, paths_file
