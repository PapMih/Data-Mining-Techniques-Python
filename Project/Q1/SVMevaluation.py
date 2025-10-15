import pandas as pd
import time

# Text processing imports


# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

# Load train and test datasets with random sampling
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
vectorizer = TfidfVectorizer(stop_words="english")
X_train_bow = vectorizer.fit_transform(df_train['FullText'])
Y_train = df_train['Label']

num_features = len(vectorizer.get_feature_names_out())
print(f"Total number of features (words in dictionary): {num_features}")
print("End data processing")

# --- Train & Evaluate Models ---
X_train, X_test, Y_train, Y_test = train_test_split(X_train_bow, Y_train, test_size=0.2, random_state=42,
                                                    stratify=Y_train)

c_values = [0.1, 1, 10, 100]  # Different C values to test
results = {}
timing_results = {}

for C_val in c_values:
    print(f"Training LinearSVC with C={C_val}")
    start_time = time.time()

    model = LinearSVC(C=C_val, max_iter=10000)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    end_time = time.time()

    elapsed_time = end_time - start_time
    timing_results[C_val] = elapsed_time

    accuracy = accuracy_score(Y_test, Y_pred)
    results[C_val] = accuracy

    print(f"Accuracy for C={C_val}: {accuracy:.4f}")
    print(f"Time taken for C={C_val}: {elapsed_time:.2f} seconds")
    print(classification_report(Y_test, Y_pred))
    print("-" * 50)

best_c = max(results, key=results.get)
print(f"Best C value: {best_c} with accuracy: {results[best_c]:.4f}")
print(f"Time taken for best C={best_c}: {timing_results[best_c]:.2f} seconds")
