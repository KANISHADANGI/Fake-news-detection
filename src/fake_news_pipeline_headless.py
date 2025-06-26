#!/usr/bin/env python3
"""
Fake News Detection ML Pipeline (Headless Version)
This version saves plots instead of displaying them interactively
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV

# Text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Statistical analysis
from scipy import stats
import joblib
import os

class FakeNewsDetector:
    def __init__(self, data_path="../TruthSeeker2023/"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.models = {}
        self.best_model = None
        self.best_threshold = 0.5
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self):
        """Step 1: Load and initial examination of data"""
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)
        
        # Load the main dataset
        print("Loading Truth_Seeker_Model_Dataset.csv...")
        self.df_main = pd.read_csv(f"{self.data_path}Truth_Seeker_Model_Dataset.csv")
        
        # Load the features dataset
        print("Loading Features_For_Traditional_ML_Techniques.csv...")
        self.df_features = pd.read_csv(f"{self.data_path}Features_For_Traditional_ML_Techniques.csv")
        
        print(f"Main dataset shape: {self.df_main.shape}")
        print(f"Features dataset shape: {self.df_features.shape}")
        print(f"Main dataset columns: {list(self.df_main.columns)}")
        print(f"Features dataset columns: {list(self.df_features.columns[:10])}...")
        
        return self.df_main, self.df_features
    
    def clean_data(self):
        """Step 2: Data Cleaning"""
        print("\n" + "=" * 60)
        print("STEP 2: DATA CLEANING")
        print("=" * 60)
        
        # Clean main dataset
        print("Cleaning main dataset...")
        initial_shape = self.df_main.shape
        
        # Remove rows with missing target values
        self.df_main = self.df_main.dropna(subset=['BinaryNumTarget'])
        
        # Clean statement text
        self.df_main = self.df_main.dropna(subset=['statement'])
        self.df_main['statement'] = self.df_main['statement'].astype(str)
        
        # Remove duplicates
        self.df_main = self.df_main.drop_duplicates(subset=['statement'])
        
        print(f"Main dataset: {initial_shape} -> {self.df_main.shape}")
        
        # Clean features dataset
        print("Cleaning features dataset...")
        initial_shape = self.df_features.shape
        
        # Remove rows with missing target values
        self.df_features = self.df_features.dropna(subset=['BinaryNumTarget'])
        
        # Handle missing values in numerical features
        numerical_cols = self.df_features.select_dtypes(include=[np.number]).columns
        
        # Fill missing values with median for numerical columns
        for col in numerical_cols:
            if col not in ['BinaryNumTarget']:
                self.df_features[col] = self.df_features[col].fillna(self.df_features[col].median())
        
        print(f"Features dataset: {initial_shape} -> {self.df_features.shape}")
        
        # Check for missing values
        print(f"Missing values in main dataset: {self.df_main.isnull().sum().sum()}")
        print(f"Missing values in features dataset: {self.df_features.isnull().sum().sum()}")
        
        return self.df_main, self.df_features
    
    def preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_data(self):
        """Step 3: Data Preprocessing"""
        print("\n" + "=" * 60)
        print("STEP 3: DATA PREPROCESSING")
        print("=" * 60)
        
        # Preprocess text in main dataset
        print("Preprocessing text data...")
        self.df_main['cleaned_statement'] = self.df_main['statement'].apply(self.preprocess_text)
        
        # Handle tweet data if available
        if 'tweet' in self.df_main.columns:
            self.df_main['cleaned_tweet'] = self.df_main['tweet'].astype(str).apply(self.preprocess_text)
        
        # Combine text features
        if 'cleaned_tweet' in self.df_main.columns:
            self.df_main['combined_text'] = (self.df_main['cleaned_statement'] + ' ' + 
                                           self.df_main['cleaned_tweet'])
        else:
            self.df_main['combined_text'] = self.df_main['cleaned_statement']
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        text_features = self.tfidf_vectorizer.fit_transform(self.df_main['combined_text'])
        
        # Convert to DataFrame
        feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        tfidf_df = pd.DataFrame(text_features.toarray(), columns=feature_names, 
                               index=self.df_main.index)
        
        # Merge with main dataset
        self.df_main_processed = pd.concat([self.df_main, tfidf_df], axis=1)
        
        # Prepare features dataset
        print("Preprocessing features dataset...")
        # Remove non-numerical columns except target
        feature_cols = self.df_features.select_dtypes(include=[np.number]).columns.tolist()
        if 'BinaryNumTarget' in feature_cols:
            feature_cols.remove('BinaryNumTarget')
        
        # Scale numerical features
        self.df_features_scaled = self.df_features.copy()
        self.df_features_scaled[feature_cols] = self.scaler.fit_transform(
            self.df_features[feature_cols])
        
        print(f"Text features shape: {text_features.shape}")
        print(f"Traditional features count: {len(feature_cols)}")
        
        return self.df_main_processed, self.df_features_scaled
    
    def exploratory_data_analysis(self):
        """Step 4: Exploratory Data Analysis"""
        print("\n" + "=" * 60)
        print("STEP 4: EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Create output directory for plots
        os.makedirs('eda_plots', exist_ok=True)
        
        # Basic statistics
        print("Dataset Statistics:")
        print(f"Total samples in main dataset: {len(self.df_main)}")
        print(f"Total samples in features dataset: {len(self.df_features)}")
        
        # Target distribution
        print("\nTarget Distribution (Main Dataset):")
        target_counts = self.df_main['BinaryNumTarget'].value_counts()
        print(target_counts)
        print(f"Fake news percentage: {(target_counts[0] / len(self.df_main)) * 100:.2f}%")
        print(f"Real news percentage: {(target_counts[1] / len(self.df_main)) * 100:.2f}%")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Target distribution
        self.df_main['BinaryNumTarget'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Target Distribution (Main Dataset)')
        axes[0,0].set_xlabel('Label (0=Fake, 1=Real)')
        axes[0,0].set_ylabel('Count')
        
        # Text length distribution
        self.df_main['statement_length'] = self.df_main['statement'].str.len()
        axes[0,1].hist(self.df_main['statement_length'], bins=50, alpha=0.7)
        axes[0,1].set_title('Statement Length Distribution')
        axes[0,1].set_xlabel('Character Count')
        axes[0,1].set_ylabel('Frequency')
        
        # Features dataset target distribution
        if len(self.df_features) > 0:
            self.df_features['BinaryNumTarget'].value_counts().plot(kind='bar', ax=axes[1,0])
            axes[1,0].set_title('Target Distribution (Features Dataset)')
            axes[1,0].set_xlabel('Label (0=Fake, 1=Real)')
            axes[1,0].set_ylabel('Count')
        
        # Correlation heatmap of top features
        if len(self.df_features) > 0:
            numerical_cols = self.df_features.select_dtypes(include=[np.number]).columns[:10]
            corr_matrix = self.df_features[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,1])
            axes[1,1].set_title('Feature Correlation Matrix (Top 10)')
        
        plt.tight_layout()
        plt.savefig('eda_plots/overview_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Saved overview analysis plot to eda_plots/overview_analysis.png")
        
        # Word cloud for fake vs real news
        try:
            from wordcloud import WordCloud
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            fake_text = ' '.join(self.df_main[self.df_main['BinaryNumTarget'] == 0]['cleaned_statement'].dropna())
            real_text = ' '.join(self.df_main[self.df_main['BinaryNumTarget'] == 1]['cleaned_statement'].dropna())
            
            if fake_text:
                wordcloud_fake = WordCloud(width=400, height=400, background_color='white').generate(fake_text)
                axes[0].imshow(wordcloud_fake, interpolation='bilinear')
                axes[0].set_title('Fake News Word Cloud')
                axes[0].axis('off')
            
            if real_text:
                wordcloud_real = WordCloud(width=400, height=400, background_color='white').generate(real_text)
                axes[1].imshow(wordcloud_real, interpolation='bilinear')
                axes[1].set_title('Real News Word Cloud')
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig('eda_plots/wordclouds.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved word clouds plot to eda_plots/wordclouds.png")
            
        except ImportError:
            print("WordCloud not available. Skipping word cloud generation.")
        
        return target_counts
    
    def prepare_datasets(self):
        """Prepare datasets with 70/20/10 split"""
        print("\n" + "=" * 60)
        print("STEP 5: PREPARING DATASETS (70/20/10 SPLIT)")
        print("=" * 60)
        
        # Prepare features and target for main dataset
        feature_cols = [col for col in self.df_main_processed.columns if col.startswith('tfidf_')]
        X_main = self.df_main_processed[feature_cols]
        y_main = self.df_main_processed['BinaryNumTarget']
        
        # Prepare features and target for traditional ML dataset
        if len(self.df_features_scaled) > 0:
            numerical_cols = self.df_features_scaled.select_dtypes(include=[np.number]).columns.tolist()
            if 'BinaryNumTarget' in numerical_cols:
                numerical_cols.remove('BinaryNumTarget')
            
            X_features = self.df_features_scaled[numerical_cols]
            y_features = self.df_features_scaled['BinaryNumTarget']
            
            # Use features dataset as primary
            X = X_features
            y = y_features
            dataset_type = "Traditional ML Features"
        else:
            X = X_main
            y = y_main
            dataset_type = "TF-IDF Features"
        
        print(f"Using {dataset_type} dataset")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # 70/20/10 split
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Second split: 20% validation, 10% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp
        )
        
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        
        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def model_selection(self):
        """Step 6: Model Selection and Training"""
        print("\n" + "=" * 60)
        print("STEP 6: MODEL SELECTION")
        print("=" * 60)
        
        # Define models to test (removed SVM for speed and Gradient Boosting)
        base_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Naive Bayes': MultinomialNB()
        }

        models = {name: CalibratedClassifierCV(base_model, method='isotonic', cv=3) 
                  for name, base_model in base_models.items()}
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Predict on training set for overfitting check
                y_train_pred = model.predict(self.X_train)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)

                # Predict on validation set
                y_pred_proba = model.predict_proba(self.X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                best_threshold = 0.5
                if y_pred_proba is not None:
                    precisions, recalls, thresholds = precision_recall_curve(self.y_val, y_pred_proba)
                    # To avoid division by zero, we can filter out the case where precision and recall are both zero.
                    fscores = np.divide(2 * precisions * recalls, precisions + recalls, 
                                        out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
                    best_threshold = thresholds[np.argmax(fscores)]
                    print(f"  Best threshold found: {best_threshold:.4f}")

                # Predict with the best threshold
                y_pred = (y_pred_proba >= best_threshold).astype(int) if y_pred_proba is not None else model.predict(self.X_val)

                # Calculate metrics
                accuracy = accuracy_score(self.y_val, y_pred)
                precision = precision_score(self.y_val, y_pred)
                recall = recall_score(self.y_val, y_pred)
                f1 = f1_score(self.y_val, y_pred)
                
                # Check for overfitting (e.g., if training accuracy is >5% higher than validation)
                overfitting = train_accuracy > (accuracy + 0.05)
                if overfitting:
                    print(f"  ⚠️  WARNING: Potential overfitting detected for {name}.")
                    print(f"      Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {accuracy:.4f}")

                if y_pred_proba is not None:
                    auc = roc_auc_score(self.y_val, y_pred_proba)
                else:
                    auc = 0
                
                model_scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'model': model,
                    'threshold': best_threshold,
                    'overfitting': overfitting
                }
                
                print(f"  Validation Accuracy: {accuracy:.4f}")
                print(f"  F1-score: {f1:.4f}")
                
            except Exception as e:
                print(f"Could not train {name}. Error: {e}")
        
        self.model_scores = model_scores
        
        # Select best model
        if not self.model_scores:
            print("No models were successfully trained.")
            self.best_model = None
            return

        # Find the best model based on F1-score, avoiding overfitting models
        sorted_models = sorted(self.model_scores.items(), key=lambda item: item[1]['f1'], reverse=True)
        
        best_model_name = None
        for name, scores in sorted_models:
            if not scores.get('overfitting', False):
                best_model_name = name
                break
        
        # If all models are overfitting, pick the best one anyway with a warning
        if best_model_name is None:
            best_model_name = sorted_models[0][0]
            print("  ⚠️  WARNING: All models show signs of overfitting. Selecting the best F1 score regardless.")

        print(f"\nBest model selected: {best_model_name} (F1-score: {self.model_scores[best_model_name]['f1']:.4f})")
        self.best_model = self.model_scores[best_model_name]['model']
        self.best_threshold = self.model_scores[best_model_name]['threshold']
    
    def evaluate_model(self):
        """Step 7: Model Evaluation"""
        print("\n" + "=" * 60)
        print("STEP 7: MODEL EVALUATION")
        print("=" * 60)
        
        if self.best_model is None:
            print("No trained model available for evaluation")
            return {}
        
        # Predictions on test set
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        y_pred = (y_pred_proba >= self.best_threshold).astype(int) if y_pred_proba is not None else self.best_model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print("Final Model Performance on Test Set:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        auc = None
        if y_pred_proba is not None:
            auc = roc_auc_score(self.y_test, y_pred_proba)
            print(f"AUC-ROC: {auc:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Fake', 'Real']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticklabels(['Fake', 'Real'])
        axes[0].set_yticklabels(['Fake', 'Real'])
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No probability scores available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ROC Curve - N/A')
        
        plt.tight_layout()
        plt.savefig('eda_plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved model evaluation plot to eda_plots/model_evaluation.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def predict_new_data(self, text_input=None):
        """Step 8: Predict on new data"""
        print("\n" + "=" * 60)
        print("STEP 8: PREDICTION ON NEW DATA")
        print("=" * 60)
        
        if self.best_model is None:
            print("No trained model available for prediction")
            return None
        
        if text_input is None:
            # Use the entire test set for a comprehensive evaluation
            print("Evaluating on the full test set...")
            X_sample = self.X_test
            y_true = self.y_test.values

            y_pred_proba = self.best_model.predict_proba(X_sample) if hasattr(self.best_model, 'predict_proba') else None
            y_pred = (y_pred_proba[:, 1] >= self.best_threshold).astype(int) if y_pred_proba is not None else self.best_model.predict(X_sample)
            
            print("\n--- Test Set Evaluation Summary ---")
            correct_predictions = np.sum(y_pred == y_true)
            total_predictions = len(y_true)
            print(f"Correctly Classified: {correct_predictions}/{total_predictions} ({correct_predictions/total_predictions:.2%})")

            print("\nSample Predictions from Test Set (with confidence):")
            num_samples_to_show = min(10, len(X_sample))
            sample_indices = np.random.choice(len(X_sample), num_samples_to_show, replace=False)

            for i, idx in enumerate(sample_indices):
                pred = y_pred[idx]
                true = y_true[idx]
                
                # Show confidence of the predicted class
                prob = f"{y_pred_proba[idx, pred]:.3f}" if y_pred_proba is not None else "N/A"
                status = "✓" if pred == true else "✗"
                label_pred = "Real" if pred == 1 else "Fake"
                label_true = "Real" if true == 1 else "Fake"
                print(f"Sample {i+1}: Predicted={label_pred} (Actual={label_true}) {status} - Confidence: {prob}")

            return y_pred

        else:
            # This part is for single-text input, which we can implement later.
            print("Prediction for new text input is not fully implemented in this version.")
            return None
    
    def save_model(self, filename='fake_news_model.pkl'):
        """Save the trained model"""
        if self.best_model is None:
            print("No trained model to save")
            return
            
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")

    def run_pipeline(self):
        """Run the entire fake news detection pipeline"""
        self.load_data()
        self.clean_data()
        self.preprocess_data()
        self.exploratory_data_analysis()
        self.prepare_datasets()
        self.model_selection()
        self.evaluate_model()
        self.predict_new_data()
        self.save_model()

if __name__ == '__main__':
    detector = FakeNewsDetector()
    results = detector.run_pipeline()