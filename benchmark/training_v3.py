"""
Simplified ML Pipeline - Direct CSV Training
Author: ML Pipeline
Date: 2025
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# ML Libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, classification_report
)

# Configuration
CONFIG = {
    'BASE_DIR': '/kaggle/working/FASTA-KmerReduce/rki_2025/model',
    'RANDOM_STATE': 42
}


class SimpleDataLoader:
    """Simple Data Loader untuk load data langsung dari CSV"""
    
    def __init__(self):
        pass
    
    def load_from_csv(self, X_train_path, X_test_path, y_train_path, y_test_path, 
                     label_encoder_path=None):
        """
        Load data dari CSV files
        
        Parameters:
        -----------
        X_train_path : str
            Path ke X_train CSV file
        X_test_path : str
            Path ke X_test CSV file
        y_train_path : str
            Path ke y_train CSV/NPY file
        y_test_path : str
            Path ke y_test CSV/NPY file
        label_encoder_path : str, optional
            Path ke label encoder pickle file
            
        Returns:
        --------
        dict : Dictionary dengan X_train, X_test, y_train, y_test, label_encoder
        """
        print("\n🔍 Loading dataset from CSV files...")
        print(f"   X_train: {X_train_path}")
        print(f"   X_test: {X_test_path}")
        print(f"   y_train: {y_train_path}")
        print(f"   y_test: {y_test_path}")
        
        try:
            # Load features
            print("\n📁 Loading features...")
            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            
            print(f"✅ X_train loaded: {X_train.shape}")
            print(f"✅ X_test loaded: {X_test.shape}")
            
            # Load labels (support both CSV and NPY)
            print("\n🏷️  Loading labels...")
            if y_train_path.endswith('.npy'):
                y_train = np.load(y_train_path)
            elif y_train_path.endswith('.csv'):
                y_train_df = pd.read_csv(y_train_path)
                # Try different column names
                if 'y_train' in y_train_df.columns:
                    y_train = y_train_df['y_train'].values
                elif 'label' in y_train_df.columns:
                    y_train = y_train_df['label'].values
                else:
                    y_train = y_train_df.iloc[:, 0].values
            else:
                raise ValueError(f"Unsupported file format for y_train: {y_train_path}")
            
            if y_test_path.endswith('.npy'):
                y_test = np.load(y_test_path)
            elif y_test_path.endswith('.csv'):
                y_test_df = pd.read_csv(y_test_path)
                if 'y_test' in y_test_df.columns:
                    y_test = y_test_df['y_test'].values
                elif 'label' in y_test_df.columns:
                    y_test = y_test_df['label'].values
                else:
                    y_test = y_test_df.iloc[:, 0].values
            else:
                raise ValueError(f"Unsupported file format for y_test: {y_test_path}")
            
            print(f"✅ y_train loaded: {y_train.shape}")
            print(f"✅ y_test loaded: {y_test.shape}")
            
            # Load label encoder if provided
            label_encoder = None
            if label_encoder_path and os.path.exists(label_encoder_path):
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                print(f"✅ Label encoder loaded: {len(label_encoder.classes_)} classes")
            
            # Validate dimensions
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"Dimension mismatch: X_train({X_train.shape[0]}) vs y_train({y_train.shape[0]})")
            
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(f"Dimension mismatch: X_test({X_test.shape[0]}) vs y_test({y_test.shape[0]})")
            
            # Select only component/feature columns
            feature_cols = [col for col in X_train.columns 
                          if col.startswith('component_') or col.startswith('feature_')]
            
            if not feature_cols:
                # If no component columns, use all numeric columns
                feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            print(f"\n📊 Using {len(feature_cols)} features")
            
            X_train = X_train[feature_cols].fillna(0)
            X_test = X_test[feature_cols].fillna(0)
            
            result = {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'y_train': y_train,
                'y_test': y_test,
                'label_encoder': label_encoder,
                'feature_names': feature_cols
            }
            
            print(f"\n✅ Data loaded successfully!")
            print(f"   X_train: {result['X_train'].shape}")
            print(f"   X_test: {result['X_test'].shape}")
            print(f"   y_train: {result['y_train'].shape}")
            print(f"   y_test: {result['y_test'].shape}")
            print(f"   Unique classes: {len(np.unique(y_train))}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_ml_pipelines():
    """Create ML pipelines with preprocessing - FIXED VERSION"""
    
    # Define models with StandardScaler in pipeline
    models = {
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'NaiveBayes': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=CONFIG['RANDOM_STATE'],
                n_jobs=-1
            ))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=CONFIG['RANDOM_STATE'],
                n_jobs=-1,
                eval_metric='mlogloss'
            ))
        ])
    }
    
    return models


def evaluate_model(model, X_test, y_test, model_name, label_encoder=None):
    """Evaluate model and return metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall_Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'Precision_Macro': precision_score(y_test, y_pred, average='macro', zero_division=0)
    }
    
    # ROC AUC (if probabilities available)
    if hasattr(model.named_steps['classifier'], 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_test)
            if len(np.unique(y_test)) > 2:
                metrics['ROC_AUC'] = roc_auc_score(y_test, y_prob, 
                                                   multi_class='ovr', 
                                                   average='macro')
            else:
                metrics['ROC_AUC'] = roc_auc_score(y_test, y_prob[:, 1])
        except Exception as e:
            print(f"   ⚠️ Could not calculate ROC AUC: {e}")
            metrics['ROC_AUC'] = np.nan
    else:
        metrics['ROC_AUC'] = np.nan
    
    print(f"   ✅ Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   ✅ F1 Score: {metrics['F1_Macro']:.4f}")
    if not np.isnan(metrics['ROC_AUC']):
        print(f"   ✅ ROC AUC: {metrics['ROC_AUC']:.4f}")
    
    return metrics, y_pred


def run_training_pipeline(X_train, X_test, y_train, y_test, 
                         dataset_name='dataset', 
                         label_encoder=None,
                         output_dir=None):
    """
    Run complete training pipeline
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    dataset_name : str
        Name for this dataset (used in output files)
    label_encoder : object, optional
        Label encoder for converting predictions back to original labels
    output_dir : str, optional
        Directory to save models and results
        
    Returns:
    --------
    tuple : (results_df, best_model_name, best_model_pipeline)
    """
    print(f"\n🚀 **TRAINING PIPELINE**")
    print(f"Dataset: {dataset_name}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Classes: {len(np.unique(y_train))}")
    print("="*60)
    
    # Create output directory
    if output_dir is None:
        output_dir = CONFIG['BASE_DIR']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipelines
    models = create_ml_pipelines()
    
    results = []
    all_predictions = []
    trained_models = {}
    
    for model_name, pipeline in tqdm(models.items(), desc="Training models"):
        model_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_model.pkl")
        
        # Training
        if not os.path.exists(model_path):
            print(f"\n⏳ Training {model_name}...")
            start_time = time.time()
            
            try:
                pipeline.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Save model
                joblib.dump(pipeline, model_path)
                print(f"✅ Model saved: {model_path}")
                
                trained_models[model_name] = pipeline
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        else:
            print(f"\n📂 Loading {model_name}...")
            try:
                pipeline = joblib.load(model_path)
                train_time = 0
                trained_models[model_name] = pipeline
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                continue
        
        # Evaluation
        start_pred_time = time.time()
        try:
            metrics, y_pred = evaluate_model(pipeline, X_test, y_test, 
                                           model_name, label_encoder)
            pred_time = time.time() - start_pred_time
            
            metrics['Train_Time_Seconds'] = round(train_time, 3)
            metrics['Predict_Time_Seconds'] = round(pred_time, 3)
            
            results.append(metrics)
            
            # Store predictions
            pred_df = pd.DataFrame({
                'Model': f"{dataset_name}_{model_name}",
                'True_Label': y_test,
                'Pred_Label': y_pred
            })
            
            # Add original labels if encoder available
            if label_encoder is not None:
                try:
                    pred_df['True_Label_Original'] = label_encoder.inverse_transform(y_test)
                    pred_df['Pred_Label_Original'] = label_encoder.inverse_transform(y_pred)
                except:
                    pass
            
            all_predictions.append(pred_df)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        gc.collect()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(output_dir, f"{dataset_name}_evaluation.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved: {results_path}")
        
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            pred_path = os.path.join(output_dir, f"{dataset_name}_predictions.csv")
            predictions_df.to_csv(pred_path, index=False)
            print(f"✅ Predictions saved: {pred_path}")
        
        # Print summary
        print(f"\n📊 **TRAINING COMPLETED**")
        print("="*60)
        print(results_df.sort_values('F1_Macro', ascending=False).to_string(index=False))
        
        best_model_row = results_df.sort_values('F1_Macro', ascending=False).iloc[0]
        best_model_name = best_model_row['Model']
        
        print(f"\n🏆 Best Model: {best_model_name} (F1: {best_model_row['F1_Macro']:.4f})")
        
        # Save best model separately
        if best_model_name in trained_models:
            best_model_path = os.path.join(output_dir, f"{dataset_name}_BEST_MODEL.pkl")
            joblib.dump(trained_models[best_model_name], best_model_path)
            print(f"✅ Best model saved: {best_model_path}")
        
        return results_df, best_model_name, trained_models.get(best_model_name)
    else:
        print("❌ No models were successfully trained")
        return None, None, None


# ================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    """
    Example usage for training without reduction_log
    """
    
    print("🚀 **SIMPLE TRAINING PIPELINE - EXAMPLE USAGE**")
    print("="*60)
    
    # Example 1: Basic usage
    print("\n1️⃣ **BASIC USAGE**")
    print("-" * 30)
    
    # Initialize loader
    loader = SimpleDataLoader()
    
    # Load your data
    data = loader.load_from_csv(
        X_train_path="/path/to/X_train.csv",
        X_test_path="/path/to/X_test.csv",
        y_train_path="/path/to/y_train.csv",  # or .npy
        y_test_path="/path/to/y_test.csv",    # or .npy
        label_encoder_path="/path/to/label_encoder.pkl"  # optional
    )
    
    if data is not None:
        # Run training
        results_df, best_model_name, best_model = run_training_pipeline(
            X_train=data['X_train'],
            X_test=data['X_test'],
            y_train=data['y_train'],
            y_test=data['y_test'],
            dataset_name='my_dataset',
            label_encoder=data['label_encoder'],
            output_dir='/kaggle/working/FASTA-KmerReduce/rki_2025/model'
        )
    
    # Example 2: Direct usage with numpy arrays
    print("\n2️⃣ **DIRECT USAGE WITH NUMPY ARRAYS**")
    print("-" * 30)
    print("""
    # If you already have numpy arrays:
    results_df, best_model_name, best_model = run_training_pipeline(
        X_train=X_train_array,
        X_test=X_test_array,
        y_train=y_train_array,
        y_test=y_test_array,
        dataset_name='my_dataset',
        output_dir='/kaggle/working/FASTA-KmerReduce/rki_2025/model'
    )
    """)
    
    # Example 3: Multiple datasets
    print("\n3️⃣ **BATCH PROCESSING MULTIPLE DATASETS**")
    print("-" * 30)
    print("""
    loader = SimpleDataLoader()
    
    datasets = [
        {
            'name': 'species_k6',
            'X_train': '/path/to/species_k6_X_train.csv',
            'X_test': '/path/to/species_k6_X_test.csv',
            'y_train': '/path/to/species_k6_y_train.npy',
            'y_test': '/path/to/species_k6_y_test.npy'
        },
        {
            'name': 'genus_k8',
            'X_train': '/path/to/genus_k8_X_train.csv',
            'X_test': '/path/to/genus_k8_X_test.csv',
            'y_train': '/path/to/genus_k8_y_train.npy',
            'y_test': '/path/to/genus_k8_y_test.npy'
        }
    ]
    
    all_results = []
    best_models = []
    
    for ds in datasets:
        data = loader.load_from_csv(
            X_train_path=ds['X_train'],
            X_test_path=ds['X_test'],
            y_train_path=ds['y_train'],
            y_test_path=ds['y_test']
        )
        
        if data:
            results_df, best_name, best_model = run_training_pipeline(
                X_train=data['X_train'],
                X_test=data['X_test'],
                y_train=data['y_train'],
                y_test=data['y_test'],
                dataset_name=ds['name']
            )
            
            if results_df is not None:
                all_results.append(results_df)
                best_models.append({
                    'dataset': ds['name'],
                    'model_name': best_name,
                    'model': best_model
                })
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv('/kaggle/working/FASTA-KmerReduce/rki_2025/model/all_results.csv', index=False)
        print(f"✅ Combined results saved")
    """)