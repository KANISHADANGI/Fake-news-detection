# Fake News Detection

This project implements a machine learning pipeline to detect fake news. It uses textual content from news statements and associated tweets, along with other engineered features, to train a classification model.

## Project Structure

```
.
├── TruthSeeker2023/
│   ├── Features_For_Traditional_ML_Techniques.csv
│   ├── readme.txt
│   └── Truth_Seeker_Model_Dataset.csv
├── eda_plots/
│   ├── model_evaluation.png
│   ├── overview_analysis.png
│   └── wordclouds.png
├── src/
│   ├── fake_news_pipeline_headless.py
│   └── fake_news_model.pkl
├── .gitignore
├── README.md
└── requirements.txt
```

-   **`TruthSeeker2023/`**: Contains the datasets used for training and evaluation.
    -   `Truth_Seeker_Model_Dataset.csv`: The main dataset with statements, tweets, and truth labels.
    -   `Features_For_Traditional_ML_Techniques.csv`: A dataset with engineered features.
    -   `readme.txt`: Provides a detailed description of the columns in the datasets.
-   **`eda_plots/`**: Contains plots generated during the Exploratory Data Analysis (EDA) phase, such as target distribution and feature correlations.
-   **`src/`**: Contains the source code for the project.
    -   `fake_news_pipeline_headless.py`: The main Python script that runs the entire machine learning pipeline from data loading to model saving.
    -   `fake_news_model.pkl`: The serialized, trained machine learning model.
-   **`requirements.txt`**: A list of Python packages required to run this project.
-   **`README.md`**: This file.

## Dataset

The data is from the `TruthSeeker2023` dataset. It includes two main files:

1.  **`Truth_Seeker_Model_Dataset.csv`**: Contains the core text data, including the news `statement` and associated `tweet` text, along with a `BinaryNumTarget` label (1 for True, 0 for False).
2.  **`Features_For_Traditional_ML_Techniques.csv`**: Provides a rich set of pre-engineered features derived from the text and user metadata, such as linguistic features (verb counts, adjectives), user metrics (followers, friends), and named entity recognition (NER) tag percentages.

For a complete description of all features, please refer to `TruthSeeker2023/readme.txt`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
## Usage

The entire pipeline can be executed by running the main script from the `src` directory.

```bash
python src/fake_news_pipeline_headless.py
```

This script will perform the following steps:
1.  **Load Data**: Loads the two CSV datasets.
2.  **Clean Data**: Handles missing values and removes duplicates.
3.  **Preprocess Data**: Cleans and preprocesses the text data (tokenization, stop-word removal, lemmatization) and creates TF-IDF features. It also scales the numerical features from the second dataset.
4.  **Exploratory Data Analysis (EDA)**: Generates and saves visualizations of the data to the `eda_plots/` directory.
5.  **Prepare Datasets**: Splits the data into training, validation, and testing sets.
6.  **Model Selection**: Trains several classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting, SVM) and uses GridSearchCV to find the best hyperparameters.
7.  **Evaluate Model**: Evaluates the best model on the test set and saves evaluation plots (confusion matrix, ROC curve) to `eda_plots/`.
8.  **Save Model**: Saves the trained and best-performing model to `src/fake_news_model.pkl`.

## Results

The pipeline evaluates various models and selects the best one based on performance metrics like F1-score and AUC. The evaluation results and plots are saved in the `eda_plots/` directory, which includes:
- `overview_analysis.png`: General EDA plots.
- `model_evaluation.png`: Performance of the final model (e.g., confusion matrix, ROC curve).
- `wordclouds.png`: Word clouds for fake and real news.
