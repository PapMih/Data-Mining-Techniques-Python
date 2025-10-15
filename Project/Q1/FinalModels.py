import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Load train dataset
df_train = pd.read_csv('train.csv')
df_train['FullText'] = df_train['Title'].fillna('') + " " + df_train['Content'].fillna('')

# --- Initial Dataset Overview ---
print("Initial Dataset Overview")
print(f"Total Train Records: {len(df_train)}")
print(df_train['Label'].value_counts())

#Remove duplicates
df_train.drop_duplicates(subset=['Id'], keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on ID: {len(df_train)}")
df_train.drop_duplicates(subset='Title', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Title: {len(df_train)}")
df_train.drop_duplicates(subset='Content', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Content: {len(df_train)}")
print(df_train['Label'].value_counts())


#StopWords
custom_stopwords = set(ENGLISH_STOP_WORDS)  # Base stopwords from Scikit-learn
custom_stopwords.update([
    "said", "say", "year", "time", "just", "make", "like", "also", "last",
    "would", "could", "may", "new", "one", "first"
])

vectorizer_svm = TfidfVectorizer(stop_words=list(custom_stopwords))

X_train_svm = vectorizer_svm.fit_transform(df_train['FullText'])
Y_train_svm = df_train['Label']
# --- Training Models --- using 5th cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- SVM with Cross-Validation ---
svm_model = LinearSVC( C=1, max_iter=10000, dual=False)
svm_accuracies = cross_val_score(svm_model, X_train_svm, Y_train_svm, cv=kf, scoring='accuracy')
svm_mean_accuracy = np.mean(svm_accuracies)
svm_std_accuracy = np.std(svm_accuracies)
print(f"\n SVM Cross-Validation Accuracy: {svm_mean_accuracy:.4f} ± {svm_std_accuracy:.4f}")

# Get predictions for confusion matrix
y_pred_svm = cross_val_predict(svm_model, X_train_svm, Y_train_svm, cv=kf)

# ---Confusion Matrix for SVM ---
cm_svm = confusion_matrix(Y_train_svm, y_pred_svm, labels=sorted(df_train['Label'].unique()))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(df_train['Label'].unique()), yticklabels=sorted(df_train['Label'].unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
plt.show()

vectorizer_rf = TfidfVectorizer(stop_words=list(custom_stopwords),max_features=20000)

X_train_rf = vectorizer_rf.fit_transform(df_train['FullText'])
Y_train_rf = df_train['Label']
# --- Random Forest with Cross-Validation ---
rf_model = RandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)
rf_accuracies = cross_val_score(rf_model, X_train_rf, Y_train_rf, cv=kf, scoring='accuracy')
rf_mean_accuracy = np.mean(rf_accuracies)
rf_std_accuracy = np.std(rf_accuracies)
print(f"\n Random Forest Cross-Validation Accuracy: {rf_mean_accuracy:.4f} ± {rf_std_accuracy:.4f}")

# Get predictions for confusion matrix
y_pred_rf = cross_val_predict(rf_model, X_train_rf, Y_train_rf, cv=kf)

# --- Confusion Matrix for Random Forest ---
cm_rf = confusion_matrix(Y_train_rf, y_pred_rf, labels=sorted(df_train['Label'].unique()))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=sorted(df_train['Label'].unique()), yticklabels=sorted(df_train['Label'].unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.show()

# --- Create a Comparative Table ---
# Create a DataFrame to summarize the results
results_summary = pd.DataFrame({
    'Statistic Measure': ['Mean Accuracy', 'Std Dev Accuracy'],
    'SVM (BoW)': [f"{svm_mean_accuracy:.4f}", f"{svm_std_accuracy:.4f}"],
    'Random Forest (BoW)': [f"{rf_mean_accuracy:.4f}", f"{rf_std_accuracy:.4f}"]
})

# Display the table
print("\nComparative Performance Table:")
print(results_summary)

# Optionally, display the table as a heatmap for visualization
plt.figure(figsize=(8, 2))
sns.heatmap(results_summary.iloc[:, 1:].astype(float), annot=True, fmt=".4f", cmap="coolwarm", cbar=False, linewidths=0.5)
plt.title("Model Performance Comparison")
plt.yticks(np.arange(2) + 0.5, results_summary['Statistic Measure'], rotation=0)
plt.show()


# Load the test dataset
df_test = pd.read_csv('test.csv')
df_test['FullText'] = df_test['Title'].fillna('') + " " + df_test['Content'].fillna('')

# Preprocess the test data using the same vectorizer as used for training
X_test_svm = vectorizer_svm.transform(df_test['FullText'])

# Train the final SVM model on the full training dataset
svm_model.fit(X_train_svm, Y_train_svm)

# Generate predictions for the test dataset
test_predictions = svm_model.predict(X_test_svm)

# Create the output DataFrame in the required format
output_df = pd.DataFrame({
    'Id': df_test['Id'],
    'Predicted': test_predictions
})

# Save the predictions to a CSV file
output_df.to_csv('testSet_categories.csv', index=False)

print("Predictions saved successfully to testSet_categories.csv")