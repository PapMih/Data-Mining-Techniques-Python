import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import nltk

nltk.download('wordnet')
nltk.download('punkt_tab')

# Load train dataset
df_train = pd.read_csv('train.csv')
df_train['FullText'] = df_train['Title'].fillna('') + " " + df_train['Content'].fillna('')


#Remove duplicates
df_train.drop_duplicates(subset=['Id'], keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on ID: {len(df_train)}")
df_train.drop_duplicates(subset='Title', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Title: {len(df_train)}")
df_train.drop_duplicates(subset='Content', keep='first', inplace=True)
print(f"Total Train Records after removing duplicates based on Content: {len(df_train)}")
print(df_train['Label'].value_counts())

#Function for text cleaning and lemmatization

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    words = word_tokenize(text)  # Tokenization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return " ".join(lemmatized_words)  # Reconstruct sentence

#  df_train['ProcessedText'] = df_train['FullText'].apply(clean_text)

#Undersampling
# #Find the minimum number of samples in any class
# min_count = df_train['Label'].value_counts().min()
# # Undersample each class to have the same number of samples as the smallest class
# df_balanced = df_train.groupby('Label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)
#
# # Check new class distribution
# print("\nClass Distribution After Undersampling:")
# print(df_balanced['Label'].value_counts())
#
#
# # ---  Dataset μετά την επεξεργασία ---
# print("\n Dataset after Processing")
# print("Dataset Shape after Preprocessing:", df_train.shape)
# print("\nFirst 5 rows after processing:\n", df_train[['ProcessedText']].head())

#StopWords
custom_stopwords = set(ENGLISH_STOP_WORDS)  # Base stopwords from Scikit-learn
custom_stopwords.update([
    "said", "say", "year", "time", "just", "make", "like", "also", "last",
    "would", "could", "may", "new", "one", "first"
])

# Use the combined list of stopwords in the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=list(custom_stopwords))

X_train_bow = vectorizer.fit_transform(df_train['FullText'])
Y_train = df_train['Label']

# --- Sample of TF-IDF Features ---
print("\n Sample of TF-IDF Vectorized Features")
num_samples = min(5, X_train_bow.shape[0])  # Ensure valid indexing
sample_indices = np.random.choice(X_train_bow.shape[0], num_samples, replace=False)
sample_features = X_train_bow[sample_indices].toarray()
sample_df = pd.DataFrame(sample_features, columns=vectorizer.get_feature_names_out())
print(sample_df.iloc[:, :10])  # Display only 10 features

print("\n TF-IDF Vocabulary & Feature Matrix")
print(f"Total number of features (words in dictionary): {X_train_bow.shape[1]}")
print(f"Feature Matrix Size (Samples x Features): {X_train_bow.shape}")
# --- Print Random Words from Vocabulary ---
num_random_words = 20  # Adjust the number of random words to print
random_words = np.random.choice(vectorizer.get_feature_names_out(), num_random_words, replace=False)

print("\n Random Words from Vocabulary:")
print(random_words)
# --- Print First 20 Words from Vocabulary ---
num_first_words = 20  # Number of words to display
first_words = vectorizer.get_feature_names_out()[:num_first_words]
print("\n First 20 Words from Vocabulary:")
print(first_words)

# --- Training Models --- using 5th cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- SVM with Cross-Validation ---
svm_model = LinearSVC( C=1.0, max_iter=10000, dual=False)
svm_accuracies = cross_val_score(svm_model, X_train_bow, Y_train, cv=kf, scoring='accuracy')
svm_mean_accuracy = np.mean(svm_accuracies)
svm_std_accuracy = np.std(svm_accuracies)
print(f"\n SVM Cross-Validation Accuracy: {svm_mean_accuracy:.4f} ± {svm_std_accuracy:.4f}")

# Get predictions for confusion matrix
y_pred_svm = cross_val_predict(svm_model, X_train_bow, Y_train, cv=kf)

# ---Confusion Matrix for SVM ---
cm_svm = confusion_matrix(Y_train, y_pred_svm, labels=sorted(df_train['Label'].unique()))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(df_train['Label'].unique()), yticklabels=sorted(df_train['Label'].unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("SVM Confusion Matrix")
plt.show()

# --- Random Forest with Cross-Validation ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_accuracies = cross_val_score(rf_model, X_train_bow, Y_train, cv=kf, scoring='accuracy')
rf_mean_accuracy = np.mean(rf_accuracies)
rf_std_accuracy = np.std(rf_accuracies)
print(f"\n Random Forest Cross-Validation Accuracy: {rf_mean_accuracy:.4f} ± {rf_std_accuracy:.4f}")

# Get predictions for confusion matrix
y_pred_rf = cross_val_predict(rf_model, X_train_bow, Y_train, cv=kf)

# --- Confusion Matrix for Random Forest ---
cm_rf = confusion_matrix(Y_train, y_pred_rf, labels=sorted(df_train['Label'].unique()))
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", xticklabels=sorted(df_train['Label'].unique()), yticklabels=sorted(df_train['Label'].unique()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Confusion Matrix")
plt.show()

# --- MOST COMMON WORDS PER CLASS ---
num_top_words = 10

# Compute mean TF-IDF across all data
mean_tfidf = np.asarray(X_train_bow.mean(axis=0)).flatten()
important_words_df = pd.DataFrame({'Word': vectorizer.get_feature_names_out(), 'Mean_TFIDF': mean_tfidf})
top_words_overall = important_words_df.sort_values(by="Mean_TFIDF", ascending=False).head(num_top_words)

print("\n Top Words Overall (Across All Categories):")
print(top_words_overall)

# Top words per class
for label in sorted(df_train['Label'].unique()):
    print(f"\n Top {num_top_words} words for label '{label}':")

    # Get indices of samples in the current category
    label_indices = (Y_train == label).to_numpy().nonzero()[0]

    # Compute mean TF-IDF for this class
    label_tfidf = X_train_bow[label_indices]
    mean_tfidf_label = np.asarray(label_tfidf.mean(axis=0)).flatten()

    # Create DataFrame and sort
    important_words_df = pd.DataFrame({'Word': vectorizer.get_feature_names_out(), 'Mean_TFIDF': mean_tfidf_label})
    top_words = important_words_df.sort_values(by="Mean_TFIDF", ascending=False).head(num_top_words)

    print(top_words)
