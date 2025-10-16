def create_project_folders(project_name='my_project',
                           dataset='dataset',
                           prep='prep',
                           model='model',
                           pred='pred',
                           path='/kaggle/working/',
                           DIR_path=None):
    import os
    """
    Membuat struktur folder proyek bioinformatika/molecular modeling.

    Parameters:
    - job_name (str): Nama utama proyek.
    - project_name (str): Nama sub-proyek atau target protein.
    - name_file (str): Nama file untuk subfolder test.
    - path (str): Root path tempat folder dibuat.

    Returns:
    - dict: Dictionary dengan path ke semua folder utama.
    """

    # Validasi nama
    invalid_chars = '^<>/\\{}[]~`$ '

    assert project_name and not set(invalid_chars).intersection(project_name), 'Invalid characters in project_name.'
    assert dataset and not set(invalid_chars).intersection(dataset), 'Invalid characters in dataset.'
    assert prep and not set(invalid_chars).intersection(prep), 'Invalid characters in prep.'
    assert model and not set(invalid_chars).intersection(model), 'Invalid characters in model.'
    assert pred and not set(invalid_chars).intersection(dataset), 'Invalid characters in pred.'

    # Tambah tanggal ke job_name
    if DIR_path == None:
        #date_str = datetime.now().strftime("%Y%m%d")
        dataset_dated = f"{dataset}" #3_{date_str}
        prep_dated = f"{prep}" #4_{date_str}
        model_dated = f"{model}" #5_{date_str}
        pred_dated = f"{pred}" #6_{date_str}

        DIR = os.path.join(path, project_name)
        DATASET_DIR = os.path.join(DIR, dataset_dated)
        PREP_DIR = os.path.join(DIR, prep_dated)
        MODEL_DIR = os.path.join(DIR, model_dated)
        PRED_DIR = os.path.join(DIR, pred_dated)

        folders = {
            'project_name': DIR,
            'dataset': DATASET_DIR,
            'prep': PREP_DIR,
            'model': MODEL_DIR,
            'pred': PRED_DIR,
        }

            #'sub_jobname': os.path.join(WRK_DIR, sub_jobname),

            # 'ML': os.path.join(WRK_DIR, f'MACHINE_LEARNING'),
            # 'DATASET_LATIH': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_LATIH'),
            # 'ANALISIS_ML': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_LATIH', 'ANALISIS_ML'),
            # 'LYPINSKI': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_LATIH', 'LYPINSKI'),
            # 'FINGERPRINT': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_LATIH', 'FINGER_PRINT'),
            # 'FP_RESULT': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_LATIH', 'FINGER_PRINT', 'FP_RESULT'),
            # 'MODEL_RESULT': os.path.join(WRK_DIR, 'MACHINE_LEARNING', 'MODEL_RESULT'),
            # 'DATASET_TEST': os.path.join(WRK_DIR, 'MACHINE_LEARNING','DATASET_TEST'),

        for name, folder in folders.items():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f'🔹 Folder {name} berhasil dibuat: {folder}')
            else:
                print(f'🔹 Folder {name} sudah dibuat: {folder}')
    else:
        DIR = DIR_path
        print(f'🔹 Folder sudah ada: {folder}')

    return folders

def ensure_dir(folder_path):
    """
    Memastikan bahwa folder tertentu ada. Jika tidak ada, folder akan dibuat.

    Parameter:
    - folder_path (str): Path folder yang ingin dipastikan keberadaannya.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Folder created: {folder_path}")
    else:
        print(f"📁 Folder already exists: {folder_path}")

def ensure_csv(folder_path, filename, data_row, columns):
    """
    Memastikan file CSV ada di folder tertentu. Jika sudah ada, tambahkan baris baru.
    Jika belum ada, buat file baru dengan kolom dan baris pertama.

    Parameters:
    - folder_path (str): Path folder tempat file berada.
    - filename (str): Nama file, misalnya 'log.csv'.
    - data_row (list or dict): Baris data yang ingin ditambahkan (dalam bentuk list atau dict).
    - columns (list): List nama kolom jika membuat file baru.
    """
    ensure_dir(folder_path)  # Pastikan folder ada
    file_path = os.path.join(folder_path, filename)

    if os.path.exists(file_path):
        # Tambahkan ke file yang sudah ada
        try:
            df_existing = pd.read_csv(file_path)
            if isinstance(data_row, dict):
                df_new = pd.DataFrame([data_row])
            else:
                df_new = pd.DataFrame([data_row], columns=columns)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(file_path, index=False)
            print(f"✅ Baris baru ditambahkan ke {file_path}")
        except Exception as e:
            print(f"❌ Gagal menambahkan ke CSV: {e}")
    else:
        # Buat file baru
        try:
            if isinstance(data_row, dict):
                df_new = pd.DataFrame([data_row])
            else:
                df_new = pd.DataFrame([data_row], columns=columns)
            df_new.to_csv(file_path, index=False)
            print(f"✅ File baru dibuat dan baris ditambahkan: {file_path}")
        except Exception as e:
            print(f"❌ Gagal membuat file CSV: {e}")
    return file_path
