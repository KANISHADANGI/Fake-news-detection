import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys

# Make sure the src directory is in the path to import the pipeline
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fake_news_pipeline_headless import FakeNewsDetector
from sklearn.ensemble import RandomForestClassifier

# Use a class-based structure for tests, similar to unittest.TestCase
class TestFakeNewsDetector:

    @pytest.fixture(scope="class")
    def detector(self):
        """Class-scoped fixture to initialize the FakeNewsDetector once."""
        return FakeNewsDetector(data_path="dummy/path/")

    @pytest.fixture(scope="class")
    def mock_main_df(self):
        """Fixture for a mock main DataFrame."""
        data = {
            'statement': [
                'This is a real news statement.', 
                'This is a fake news statement!',
                'Another real one.',
                'This is a fake news statement!', # duplicate
                None,
                'News with numbers 123.'
            ],
            'tweet': [
                'tweet for real news', 
                'tweet for fake news',
                'another tweet',
                'tweet for fake news',
                'a tweet',
                'tweet with numbers'
            ],
            'BinaryNumTarget': [1, 0, 1, 0, 1, np.nan] # one nan
        }
        return pd.DataFrame(data)

    @pytest.fixture(scope="class")
    def mock_features_df(self):
        """Fixture for a mock features DataFrame."""
        n_samples = 6
        data = {
            'followers_count': [100, 200, 150, 200, 300, 400],
            'friends_count': [50, 100, 75, 100, 150, 200],
            'statuses_count': [10, 20, 15, 20, 25, 30],
            'BinaryNumTarget': [1, 0, 1, 0, 1, 0],
            'some_other_feature': [1.1, 2.2, np.nan, 2.2, 4.4, 5.5] # with nan
        }
        return pd.DataFrame(data)

    def test_preprocess_text(self, detector):
        """Test the text preprocessing function."""
        text = "This is a Test statement with stop words, numbers 123 and Punctuations! It should be cleaned."
        # A simple mock for lemmatizer to ensure predictability
        detector.lemmatizer.lemmatize = lambda x: x
        processed_text = detector.preprocess_text(text)
        
        # Split into words to check for standalone words
        words = processed_text.split()
        
        assert "test" in words
        assert "statement" in words
        assert "is" not in words  # Check for a common stopword
        assert "a" not in words   # Check for another common stopword  
        assert "123" not in processed_text  # Check digits are removed
        assert "!" not in processed_text    # Check punctuation is removed

    def test_clean_data(self, detector, mock_main_df, mock_features_df):
        """Test the data cleaning method handles NaNs and duplicates."""
        # Create copies to avoid modifying the fixture for other tests
        detector.df_main = mock_main_df.copy()
        detector.df_features = mock_features_df.copy()

        assert detector.df_main.isnull().sum().sum() > 0
        assert detector.df_features.isnull().sum().sum() > 0
        assert detector.df_main.duplicated(subset=['statement']).any()

        cleaned_main, cleaned_features = detector.clean_data()

        # After cleaning: dropped row with NaN target, then row with NaN statement, then duplicate
        assert cleaned_main.shape[0] == 3
        assert not cleaned_main.duplicated(subset=['statement']).any()
        assert cleaned_features['some_other_feature'].isnull().sum() == 0
        
        median_val = mock_features_df['some_other_feature'].median()
        assert cleaned_features['some_other_feature'].iloc[2] == median_val

    def test_load_data(self, detector):
        """Test data loading by mocking pd.read_csv."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            mock_read_csv.return_value = mock_df

            df_main, df_features = detector.load_data()
            
            assert mock_read_csv.call_count == 2
            assert df_main.equals(mock_df)
            assert df_features.equals(mock_df)

    def test_prepare_datasets(self, detector):
        """Test the dataset preparation and splitting logic."""
        n_samples = 20  # Increase samples for stratification
        
        # Mock the processed dataframes
        detector.df_main_processed = pd.DataFrame({
            'BinaryNumTarget': [1, 0] * (n_samples // 2),
            'tfidf_0': [0.1] * n_samples,
            'tfidf_1': [0.2] * n_samples
        })
        detector.df_features_scaled = pd.DataFrame({
            'BinaryNumTarget': [1, 0] * (n_samples // 2), 
            'feature_1': [0.5] * n_samples,
            'feature_2': [0.7] * n_samples
        })

        detector.prepare_datasets()
        
        assert detector.X_train is not None
        assert detector.X_val is not None
        assert detector.X_test is not None
        assert detector.y_train is not None
        assert detector.y_val is not None
        assert detector.y_test is not None

        # Check that splits are not empty
        assert detector.y_test.shape[0] > 0
        assert detector.y_val.shape[0] > 0
        assert detector.y_train.shape[0] > 0

    def test_save_model(self, detector):
        """Test that the model and transformers are saved correctly."""
        mock_model = MagicMock()
        detector.best_model = mock_model
        
        with patch('joblib.dump') as mock_dump:
            detector.save_model(filename='test_model.pkl')
            
            mock_dump.assert_called_once()
            
            saved_obj = mock_dump.call_args[0][0]
            assert isinstance(saved_obj, dict)
            assert 'model' in saved_obj and 'scaler' in saved_obj and 'tfidf_vectorizer' in saved_obj
            assert saved_obj['model'] == mock_model

    def test_model_selection(self, detector):
        """Test that model selection logic runs and selects a best model."""
        # Mock training data with enough samples for CV
        n_samples = 20
        detector.X_train = pd.DataFrame({
            'feature1': range(n_samples), 
            'feature2': range(n_samples)
        })
        detector.y_train = pd.Series([0, 1] * (n_samples // 2))
        detector.X_val = pd.DataFrame({
            'feature1': range(n_samples), 
            'feature2': range(n_samples)
        })
        detector.y_val = pd.Series([0, 1] * (n_samples // 2))

        # Mock the model training and prediction to avoid actual ML computation
        with patch('sklearn.ensemble.RandomForestClassifier') as mock_rf, \
             patch('sklearn.linear_model.LogisticRegression') as mock_lr, \
             patch('sklearn.calibration.CalibratedClassifierCV') as mock_cal:
            
            mock_model = MagicMock()
            mock_model.fit.return_value = None
            mock_model.predict.return_value = np.array([0, 1] * (n_samples // 2))
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]] * (n_samples // 2))
            mock_cal.return_value = mock_model
            
            detector.model_selection()

            assert detector.best_model is not None
            assert detector.model_scores is not None
            assert len(detector.model_scores) > 0

    def test_predict_new_data(self, detector):
        """Test the prediction functionality for new text returns None as expected."""
        # The current implementation returns None for text input. This test verifies that.
        result = detector.predict_new_data("This is a test news article.")
        assert result is None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plotting_functions_run(self, mock_close, mock_savefig, detector):
        """Ensure plotting functions run without crashing."""
        # Mock data needed for plotting
        detector.df_main = pd.DataFrame({
            'BinaryNumTarget': [0, 1, 1, 0, 1],
            'statement': ["a", "b", "c", "d", "e"],  # Add statement column
            'cleaned_statement': ["a", "b", "c", "d", "e"]  # Add cleaned_statement for word cloud
        })
        detector.df_features = pd.DataFrame({
            'followers_count': [10, 20, 30, 40, 50],
            'BinaryNumTarget': [0, 1, 1, 0, 1]
        })
        
        # Setup test data for evaluate_model
        detector.X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        detector.y_test = pd.Series([0, 1])
        detector.best_threshold = 0.5
        
        # Mock model to have proper prediction methods
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_model.predict.return_value = np.array([0, 1])
        detector.best_model = mock_model

        try:
            # Test exploratory_data_analysis
            detector.exploratory_data_analysis()
            
            # Test evaluate_model (without parameters)
            detector.evaluate_model()
            
        except Exception as e:
            pytest.fail(f"Plotting functions failed with error: {e}")

        # Assert that savefig was called, confirming plot generation was attempted
        assert mock_savefig.call_count > 0 