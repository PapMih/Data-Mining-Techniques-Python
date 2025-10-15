import re
import pandas as pd
import numpy as np
import time

# Text processing imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load train dataset with random sampling
df_train = pd.read_csv('train.csv').sample(frac=1, random_state=42)

print(f"Total Train Records: {len(df_train)}")
print(df_train['Label'].value_counts())

# Combine Title and Content
df_train['FullText'] = df_train['Title'] + " " + df_train['Content']

# Remove duplicates
df_train.drop_duplicates(subset=['Id'], keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on ID: {len(df_train)}")
df_train.drop_duplicates(subset='Title', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Title: {len(df_train)}")
df_train.drop_duplicates(subset='Content', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Content: {len(df_train)}")
print(df_train['Label'].value_counts())

# --- Feature Extraction ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_bow = vectorizer.fit_transform(df_train['FullText'])
Y_train = df_train['Label']

# Get the total number of words (features) in the vocabulary
num_features = len(vectorizer.get_feature_names_out())
print(f"Total number of features (words in dictionary): {num_features}")
print("End data processing")

# --- Train & Evaluate Models ---
X_train, X_test, Y_train, Y_test = train_test_split(X_train_bow, Y_train, test_size=0.2, random_state=42,
                                                    stratify=Y_train)

n_estimators_values = [100, 500, 800, 1500, 3000]  # Different tree numbers to test
results = {}
timing_results = {}

for n in n_estimators_values:
    print(f"Training RandomForest with n_estimators={n}")

    start_time = time.time()  # Start time tracking

    # Train model
    model = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    model.fit(X_train, Y_train)

    # Predict on test set
    Y_pred = model.predict(X_test)

    # End time tracking
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    timing_results[n] = elapsed_time

    # Evaluate model
    accuracy = accuracy_score(Y_test, Y_pred)
    results[n] = accuracy  # Store results

    print(f"Accuracy for n_estimators={n}: {accuracy:.4f}")
    print(f"Time taken for n_estimators={n}: {elapsed_time:.2f} seconds")
    print(classification_report(Y_test, Y_pred))
    print("-" * 50)

# Find Best Number of Trees
best_n = max(results, key=results.get)
print(f"Best n_estimators: {best_n} with accuracy: {results[best_n]:.4f}")
print(f"Time taken for best n_estimators={best_n}: {timing_results[best_n]:.2f} seconds")
