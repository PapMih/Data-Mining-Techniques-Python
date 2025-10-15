# #Threshold Test
# ##############################################
# # Global imports
# ##############################################
# import time
# import pandas as pd
# import numpy as np
# from datasketch import MinHash, MinHashLSH
# from sklearn.feature_extraction.text import CountVectorizer
# import scipy.sparse as sp
# from heapq import heappush, heappop, heapreplace
#
# ###############################################
# # Global parameters
# ###############################################
# K = 7               # Number of nearest neighbors
# SHINGLE_SIZE = 6    # Character shingle size
# CHUNK_SIZE = 1000   # Chunk size for brute force
# NUM_PERM = 64      # Number of permutations for MinHash
# THRESHOLDS = [0.8, 0.5, 0.2]  # Thresholds to test in LSH
#
# ###############################################
# #  Brute Force function
# ###############################################
# def run_brute_force():
#
#     start_bf = time.time()
#
#     # -----------------------
#     #  Load and Preprocess Data
#     # -----------------------
#     print("Loading and Preprocessing Data (Brute Force)...")
#     train_df = pd.read_csv("train.csv", usecols=[0, 1, 2], sep=",", encoding="utf-8")
#     test_df = pd.read_csv("test_without_labels.csv", sep=",", encoding="utf-8")
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     # -----------------------
#     # Create Shingles
#     # -----------------------
#     print("Creating Shingles (Brute Force)...")
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(SHINGLE_SIZE, SHINGLE_SIZE))
#
#     train_csr = vectorizer.fit_transform(train_df['FullText'])
#     test_csr = vectorizer.transform(test_df['FullText'])
#
#     # -----------------------
#     # Pairwise Jaccard (Chunked)
#     # -----------------------
#     def pairwise_jaccard_chunked(
#             test_csr,
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=0,
#             global_counter=0
#     ):
#         T = test_csr.astype(bool).astype(int).tocsr()
#         R = train_csr.astype(bool).astype(int).tocsr()
#
#         intersection = T.dot(R.T)
#
#         test_sums = T.sum(axis=1).A1
#         train_sums = R.sum(axis=1).A1
#
#         local_knn = {}
#
#         for row_i in range(intersection.shape[0]):
#             startptr = intersection.indptr[row_i]
#             endptr = intersection.indptr[row_i + 1]
#             cols = intersection.indices[startptr:endptr]
#             vals = intersection.data[startptr:endptr]
#
#             t_sum = test_sums[row_i]
#             heap = []
#
#             for idx, train_j in enumerate(cols):
#                 union = t_sum + train_sums[train_j] - vals[idx]
#                 dist = 1.0 - (vals[idx] / union) if union > 0 else 1.0
#
#                 neg_dist = -dist
#                 if len(heap) < K:
#                     heappush(heap, (neg_dist, train_ids[train_j]))
#                 else:
#                     if neg_dist > heap[0][0]:
#                         heapreplace(heap, (neg_dist, train_ids[train_j]))
#
#             top_neighbors = [heappop(heap)[1] for _ in range(len(heap))][::-1]
#             test_id = test_ids[start_test_id + row_i]
#             local_knn[test_id] = top_neighbors
#
#             global_counter += 1
#
#         return local_knn, global_counter
#
#     # -----------------------
#     # Process in Chunks
#     # -----------------------
#     print("Starting Brute Force Jaccard computation...")
#     knn_results_bf = {}
#     train_ids = train_df['Id'].values
#     test_ids = test_df['Id'].values
#     n_test = test_csr.shape[0]
#
#     global_counter = 0
#     for start_idx in range(0, n_test, CHUNK_SIZE):
#         end_idx = min(start_idx + CHUNK_SIZE, n_test)
#         chunk_knn, global_counter = pairwise_jaccard_chunked(
#             test_csr[start_idx:end_idx],
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=start_idx,
#             global_counter=global_counter
#         )
#         knn_results_bf.update(chunk_knn)
#
#     print("Finished Brute Force Jaccard computation.")
#     total_time_bf = time.time() - start_bf
#
#     return knn_results_bf, total_time_bf
#
# ###############################################
# # LSH (MinHash) function
# ###############################################
# def run_lsh(threshold):
#     start_build = time.time()
#
#     print(f"\nLoading and preprocessing data (LSH) with threshold={threshold}...")
#     train_df = pd.read_csv("train.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#     test_df = pd.read_csv("test_without_labels.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     def get_shingles(text, shingle_size):
#         return {text[i: i + shingle_size] for i in range(len(text) - shingle_size + 1)}
#
#     def create_minhash(shingles, num_perm=128):
#         m = MinHash(num_perm=num_perm)
#         for shingle in shingles:
#             m.update(shingle.encode('utf8'))
#         return m
#
#     print("Generating MinHash signatures (LSH)...")
#     train_df["MinHash"] = None
#     for idx, row in train_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         train_df.at[idx, "MinHash"] = create_minhash(shingle_set, NUM_PERM)
#
#     test_df["MinHash"] = None
#     for idx, row in test_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         test_df.at[idx, "MinHash"] = create_minhash(shingle_set, NUM_PERM)
#
#     print("Building LSH index...")
#     lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
#     for _, row in train_df.iterrows():
#         train_id = str(row["Id"])
#         lsh.insert(train_id, row["MinHash"])
#
#     build_time = time.time() - start_build
#
#     start_query = time.time()
#     print("Finding nearest neighbors...")
#     knn_results_lsh = {}
#     for idx, row in test_df.iterrows():
#         test_id = row["Id"]
#         test_minhash = row["MinHash"]
#         candidate_ids = lsh.query(test_minhash)
#         neighbors = candidate_ids[:K]
#         knn_results_lsh[test_id] = [int(cid) for cid in neighbors]
#
#     print("Nearest neighbor search completed.")
#     query_time = time.time() - start_query
#
#     return build_time, query_time, knn_results_lsh
#
# ###############################################
# # Function to compute "fraction"
# ###############################################
# def compute_fraction_of_true_neighbors(brute_force_knn, lsh_knn):
#
#     total_correct = 0
#     total_possible = 0
#
#     for test_id, bf_neighbors in brute_force_knn.items():
#         lsh_neighbors = lsh_knn.get(test_id, [])
#         common = set(bf_neighbors).intersection(set(lsh_neighbors))
#         total_correct += len(common)
#         total_possible += len(bf_neighbors)
#
#     fraction = total_correct / total_possible if total_possible > 0 else 0.0
#     return fraction
#
# # Run Brute Force once
# bf_knn_results, bf_time = run_brute_force()
# print(f"Brute Force total time: {bf_time:.2f} sec\n")
#
# # Create a table of results
# results_table = []
#
#
# results_table.append({
#     "Type": "Brute-Force-Jaccard",
#     "BuildTime": 0.0,
#     "QueryTime": bf_time,
#     "TotalTime": bf_time,
#     "fraction of the true K most similar documents that are reported by LSH method as well": "100%",
#     "Parameters": ""
# })
#
# # Run LSH for each threshold
# for thr in THRESHOLDS:
#     build_time, query_time, lsh_knn_results = run_lsh(thr)
#     total_time = build_time + query_time
#
#     fraction = compute_fraction_of_true_neighbors(bf_knn_results, lsh_knn_results)
#     fraction_str = f"{fraction*100:.0f}%"
#
#     # Add an LSH-Jaccard row to table
#     results_table.append({
#         "Type": "LSH-Jaccard",
#         "BuildTime": round(build_time, 2),
#         "QueryTime": round(query_time, 2),
#         "TotalTime": round(total_time, 2),
#         "fraction of the true K most similar documents that are reported by LSH method as well": fraction_str,
#         "Parameters": f"Perm={NUM_PERM}, threshold={thr}"
#     })
#
# # Print the final table
# df = pd.DataFrame(results_table, columns=[
#     "Type",
#     "BuildTime",
#     "QueryTime",
#     "TotalTime",
#     "fraction of the true K most similar documents that are reported by LSH method as well",
#     "Parameters"
# ])
#
# print("\nFinal Results Table:\n")
# print(df.to_string(index=False))
# print()

# #######################################################################################################
# # Permutations Test
# ##############################################
# # Global imports
# ##############################################
# import time
# import pandas as pd
# import numpy as np
# from datasketch import MinHash, MinHashLSH
# from sklearn.feature_extraction.text import CountVectorizer
# import scipy.sparse as sp
# from heapq import heappush, heappop, heapreplace
#
# ###############################################
# # Global parameters
# ###############################################
# K = 7                # Number of nearest neighbors
# SHINGLE_SIZE = 6     # Character shingle size
# CHUNK_SIZE = 1000    # Chunk size for brute force
# THRESHOLD = 0.8      # Threshold for LSH
# PERM_VALUES = [16, 32, 64]  # Different number of permutations to try
#
# ###############################################
# # Brute Force function
# ###############################################
# def run_brute_force():
#
#     start_bf = time.time()
#
#     # -----------------------
#     # Load and Preprocess Data
#     # -----------------------
#     print("Loading and Preprocessing Data (Brute Force)...")
#     train_df = pd.read_csv("train.csv", usecols=[0, 1, 2], sep=",", encoding="utf-8", dtype={"Id": "int32"})
#     test_df = pd.read_csv("test_without_labels.csv", sep=",", encoding="utf-8", dtype={"Id": "int32"})
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     # -----------------------
#     # Create Shingles
#     # -----------------------
#     print("Creating Shingles (Brute Force)...")
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(SHINGLE_SIZE, SHINGLE_SIZE))
#
#     train_csr = vectorizer.fit_transform(train_df['FullText'])
#     test_csr = vectorizer.transform(test_df['FullText'])
#
#     # -----------------------
#     # Pairwise Jaccard (Chunked)
#     # -----------------------
#     def pairwise_jaccard_chunked(
#             test_csr,
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=0,
#             global_counter=0
#     ):
#         T = test_csr.astype(bool).astype(int).tocsr()
#         R = train_csr.astype(bool).astype(int).tocsr()
#
#         intersection = T.dot(R.T)
#
#         test_sums = T.sum(axis=1).A1
#         train_sums = R.sum(axis=1).A1
#
#         local_knn = {}
#
#         for row_i in range(intersection.shape[0]):
#             startptr = intersection.indptr[row_i]
#             endptr = intersection.indptr[row_i + 1]
#             cols = intersection.indices[startptr:endptr]
#             vals = intersection.data[startptr:endptr]
#
#             t_sum = test_sums[row_i]
#             heap = []
#
#             for idx, train_j in enumerate(cols):
#                 union = t_sum + train_sums[train_j] - vals[idx]
#                 dist = 1.0 - (vals[idx] / union) if union > 0 else 1.0
#
#                 neg_dist = -dist
#                 if len(heap) < K:
#                     heappush(heap, (neg_dist, train_ids[train_j]))
#                 else:
#                     if neg_dist > heap[0][0]:
#                         heapreplace(heap, (neg_dist, train_ids[train_j]))
#
#             top_neighbors = [heappop(heap)[1] for _ in range(len(heap))][::-1]
#             test_id = test_ids[start_test_id + row_i]
#             local_knn[test_id] = top_neighbors
#
#             global_counter += 1
#
#         return local_knn, global_counter
#
#     print("Starting Brute Force Jaccard computation...")
#     knn_results_bf = {}
#     train_ids = train_df['Id'].values
#     test_ids = test_df['Id'].values
#     n_test = test_csr.shape[0]
#
#     global_counter = 0
#     for start_idx in range(0, n_test, CHUNK_SIZE):
#         end_idx = min(start_idx + CHUNK_SIZE, n_test)
#         chunk_knn, global_counter = pairwise_jaccard_chunked(
#             test_csr[start_idx:end_idx],
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=start_idx,
#             global_counter=global_counter
#         )
#         knn_results_bf.update(chunk_knn)
#
#     print("Finished Brute Force Jaccard computation.")
#     total_time_bf = time.time() - start_bf
#
#     return knn_results_bf, total_time_bf
#
# ###############################################
# # LSH (MinHash) function
# ###############################################
# def run_lsh(num_perm):
#
#     start_build = time.time()
#
#     print(f"\nLoading and preprocessing data (LSH) with threshold={THRESHOLD}, num_perm={num_perm}...")
#     train_df = pd.read_csv("train.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#     test_df = pd.read_csv("test_without_labels.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     def get_shingles(text, shingle_size):
#         return {text[i: i + shingle_size] for i in range(len(text) - shingle_size + 1)}
#
#     def create_minhash(shingles, n_perm):
#         m = MinHash(num_perm=n_perm)
#         for shingle in shingles:
#             m.update(shingle.encode('utf8'))
#         return m
#
#     print("Generating MinHash signatures (LSH)...")
#     train_df["MinHash"] = None
#     for idx, row in train_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         train_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#
#     test_df["MinHash"] = None
#     for idx, row in test_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         test_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#
#     print("Building LSH index...")
#     lsh = MinHashLSH(threshold=THRESHOLD, num_perm=num_perm)
#     for _, row in train_df.iterrows():
#         train_id = str(row["Id"])
#         lsh.insert(train_id, row["MinHash"])
#
#     build_time = time.time() - start_build
#
#     start_query = time.time()
#     print("Finding nearest neighbors...")
#     knn_results_lsh = {}
#     for idx, row in test_df.iterrows():
#         test_id = row["Id"]
#         test_minhash = row["MinHash"]
#         candidate_ids = lsh.query(test_minhash)
#         neighbors = candidate_ids[:K]
#         knn_results_lsh[test_id] = [int(cid) for cid in neighbors]
#
#     print("Nearest neighbor search completed.")
#     query_time = time.time() - start_query
#
#     return build_time, query_time, knn_results_lsh
#
# ###############################################
# # Function to compute "fraction"
# ###############################################
# def compute_fraction_of_true_neighbors(brute_force_knn, lsh_knn):
#     """
#     Computes the fraction of the true K neighbors (found by brute force)
#     that are also present in the LSH results.
#     """
#     total_correct = 0
#     total_possible = 0
#
#     for test_id, bf_neighbors in brute_force_knn.items():
#         lsh_neighbors = lsh_knn.get(test_id, [])
#         common = set(bf_neighbors).intersection(set(lsh_neighbors))
#         total_correct += len(common)
#         total_possible += len(bf_neighbors)
#
#     fraction = total_correct / total_possible if total_possible > 0 else 0.0
#     return fraction
#
# # Run Brute Force once
# bf_knn_results, bf_time = run_brute_force()
# print(f"Brute Force total time: {bf_time:.2f} sec\n")
#
# # Create a table of results
# results_table = []
#
# # Add Brute Force row
# results_table.append({
#     "Type": "Brute-Force-Jaccard",
#     "BuildTime": 0.0,
#     "QueryTime": bf_time,
#     "TotalTime": bf_time,
#     "fraction of the true K most similar documents that are reported by LSH method as well": "100%",
#     "Parameters": ""
# })
#
# # Run LSH for each perm
# for perm in PERM_VALUES:
#     build_time, query_time, lsh_knn_results = run_lsh(perm)
#     total_time = build_time + query_time
#
#     fraction = compute_fraction_of_true_neighbors(bf_knn_results, lsh_knn_results)
#     fraction_str = f"{fraction*100:.0f}%"
#
#     # Add an LSH-Jaccard row
#     results_table.append({
#         "Type": "LSH-Jaccard",
#         "BuildTime": round(build_time, 2),
#         "QueryTime": round(query_time, 2),
#         "TotalTime": round(total_time, 2),
#         "fraction of the true K most similar documents that are reported by LSH method as well": fraction_str,
#         "Parameters": f"Perm={perm}, threshold={THRESHOLD}"
#     })
#
# # Print the final table
# df = pd.DataFrame(results_table, columns=[
#     "Type",
#     "BuildTime",
#     "QueryTime",
#     "TotalTime",
#     "fraction of the true K most similar documents that are reported by LSH method as well",
#     "Parameters"
# ])
#
# print("\nFinal Results Table:\n")
# print(df.to_string(index=False))
# print()

########################################################################################################
#Testing with More and Fewer Neighbors
###############################################
# Global imports
###############################################
import time
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from heapq import heappush, heappop, heapreplace

###############################################
# Global parameters
###############################################

SHINGLE_SIZE = 6     # Character shingle size
CHUNK_SIZE = 1000    # Chunk size for brute force
THRESHOLD = 0.5      # Fixed threshold for LSH
PERM_VALUES = [64]   # Permutations

# ###############################################
# # Brute Force function
# ###############################################
# def run_brute_force():
#     start_bf = time.time()
#
#     # -----------------------
#     # Load and Preprocess Data
#     # -----------------------
#     print("Loading and Preprocessing Data (Brute Force)...")
#     train_df = pd.read_csv("train.csv", usecols=[0, 1, 2], sep=",", encoding="utf-8", dtype={"Id": "int32"})
#     test_df = pd.read_csv("test_without_labels.csv", sep=",", encoding="utf-8", dtype={"Id": "int32"})
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     # -----------------------
#     # Create Shingles with a Common Vocabulary
#     # -----------------------
#     print("Creating Shingles (Brute Force)...")
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(SHINGLE_SIZE, SHINGLE_SIZE))
#
#     train_csr = vectorizer.fit_transform(train_df['FullText'])
#     test_csr = vectorizer.transform(test_df['FullText'])
#
#     # -----------------------
#     # Pairwise Jaccard (Chunked)
#     # -----------------------
#     def pairwise_jaccard_chunked(
#             test_csr,
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=0,
#             global_counter=0
#     ):
#         T = test_csr.astype(bool).astype(int).tocsr()
#         R = train_csr.astype(bool).astype(int).tocsr()
#
#         intersection = T.dot(R.T)
#
#         test_sums = T.sum(axis=1).A1
#         train_sums = R.sum(axis=1).A1
#
#         local_knn = {}
#
#         for row_i in range(intersection.shape[0]):
#             startptr = intersection.indptr[row_i]
#             endptr = intersection.indptr[row_i + 1]
#             cols = intersection.indices[startptr:endptr]
#             vals = intersection.data[startptr:endptr]
#
#             t_sum = test_sums[row_i]
#             heap = []
#
#             for idx, train_j in enumerate(cols):
#                 union = t_sum + train_sums[train_j] - vals[idx]
#                 dist = 1.0 - (vals[idx] / union) if union > 0 else 1.0
#
#                 neg_dist = -dist
#                 if len(heap) < K:
#                     heappush(heap, (neg_dist, train_ids[train_j]))
#                 else:
#                     if neg_dist > heap[0][0]:
#                         heapreplace(heap, (neg_dist, train_ids[train_j]))
#
#             top_neighbors = [heappop(heap)[1] for _ in range(len(heap))][::-1]
#             test_id = test_ids[start_test_id + row_i]
#             local_knn[test_id] = top_neighbors
#
#             global_counter += 1
#
#         return local_knn, global_counter
#
#     print("Starting Brute Force Jaccard computation...")
#     knn_results_bf = {}
#     train_ids = train_df['Id'].values
#     test_ids = test_df['Id'].values
#     n_test = test_csr.shape[0]
#
#     global_counter = 0
#     for start_idx in range(0, n_test, CHUNK_SIZE):
#         end_idx = min(start_idx + CHUNK_SIZE, n_test)
#         chunk_knn, global_counter = pairwise_jaccard_chunked(
#             test_csr[start_idx:end_idx],
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=start_idx,
#             global_counter=global_counter
#         )
#         knn_results_bf.update(chunk_knn)
#
#     print("Finished Brute Force Jaccard computation.")
#     total_time_bf = time.time() - start_bf
#
#     return knn_results_bf, total_time_bf
#
# ###############################################
# # LSH (MinHash) function
# ###############################################
# def run_lsh(num_perm):
#     start_build = time.time()
#
#     print(f"\nLoading and preprocessing data (LSH) with threshold={THRESHOLD}, num_perm={num_perm}...")
#     train_df = pd.read_csv("train.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#     test_df = pd.read_csv("test_without_labels.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     def get_shingles(text, shingle_size):
#         return {text[i: i + shingle_size] for i in range(len(text) - shingle_size + 1)}
#
#     def create_minhash(shingles, n_perm):
#         m = MinHash(num_perm=n_perm)
#         for shingle in shingles:
#             m.update(shingle.encode('utf8'))
#         return m
#
#     print("Generating MinHash signatures (LSH)...")
#     train_df["MinHash"] = None
#     for idx, row in train_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         train_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#
#     test_df["MinHash"] = None
#     for idx, row in test_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         test_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#
#     print("Building LSH index...")
#     lsh = MinHashLSH(threshold=THRESHOLD, num_perm=num_perm)
#     for _, row in train_df.iterrows():
#         train_id = str(row["Id"])
#         lsh.insert(train_id, row["MinHash"])
#
#     build_time = time.time() - start_build
#
#     start_query = time.time()
#     print("Finding nearest neighbors...")
#     knn_results_lsh = {}
#     for idx, row in test_df.iterrows():
#         test_id = row["Id"]
#         test_minhash = row["MinHash"]
#         candidate_ids = lsh.query(test_minhash)
#         neighbors = candidate_ids[:K]
#         knn_results_lsh[test_id] = [int(cid) for cid in neighbors]
#
#     print("Nearest neighbor search completed.")
#     query_time = time.time() - start_query
#
#     return build_time, query_time, knn_results_lsh
#
# ###############################################
# # Function to compute "fraction"
# ###############################################
# def compute_fraction_of_true_neighbors(brute_force_knn, lsh_knn):
#
#     total_correct = 0
#     total_possible = 0
#
#     for test_id, bf_neighbors in brute_force_knn.items():
#         lsh_neighbors = lsh_knn.get(test_id, [])
#         common = set(bf_neighbors).intersection(set(lsh_neighbors))
#         total_correct += len(common)
#         total_possible += len(bf_neighbors)
#
#     fraction = total_correct / total_possible if total_possible > 0 else 0.0
#     return fraction
#
# # For K = 5, 7, 9
# for K in [5, 7, 9]:
#     print("\n" + "#" * 40)
#     print(f"Running experiments for K = {K}")
#     print("#" * 40 + "\n")
#
#     # 1) Run Brute Force once
#     bf_knn_results, bf_time = run_brute_force()
#     print(f"Brute Force total time: {bf_time:.2f} sec\n")
#
#     # Create a table of results
#     results_table = []
#
#     # Add Brute Force row
#     results_table.append({
#         "Type": "Brute-Force-Jaccard",
#         "BuildTime": 0.0,
#         "QueryTime": bf_time,
#         "TotalTime": bf_time,
#         "fraction of the true K most similar documents that are reported by LSH method as well": "100%",
#         "Parameters": f"K={K}"
#     })
#
#     for perm in PERM_VALUES:
#         build_time, query_time, lsh_knn_results = run_lsh(perm)
#         total_time = build_time + query_time
#
#         fraction = compute_fraction_of_true_neighbors(bf_knn_results, lsh_knn_results)
#         fraction_str = f"{fraction*100:.0f}%"  # π.χ. "80%"
#
#         # Add an LSH-Jaccard row
#         results_table.append({
#             "Type": "LSH-Jaccard",
#             "BuildTime": round(build_time, 2),
#             "QueryTime": round(query_time, 2),
#             "TotalTime": round(total_time, 2),
#             "fraction of the true K most similar documents that are reported by LSH method as well": fraction_str,
#             "Parameters": f"K={K}, Perm={perm}, threshold={THRESHOLD}"
#         })
#
#     # Print the final table for the current K value
#     df = pd.DataFrame(results_table, columns=[
#         "Type",
#         "BuildTime",
#         "QueryTime",
#         "TotalTime",
#         "fraction of the true K most similar documents that are reported by LSH method as well",
#         "Parameters"
#     ])
#
#     print("\nFinal Results Table:\n")
#     print(df.to_string(index=False))
#     print()
#

# ###################################################################################################
# #Combination of LSH and Brute Force
# ########################################################################################################
#
# import time
# import pandas as pd
# import numpy as np
# from datasketch import MinHash, MinHashLSH
# from sklearn.feature_extraction.text import CountVectorizer
# import scipy.sparse as sp
# from heapq import heappush, heappop, heapreplace
#
# ###############################################
# # Global parameters
# ###############################################
# SHINGLE_SIZE = 6     # Size of each character shingle
# CHUNK_SIZE = 1000    # Chunk size for the brute force computation
# THRESHOLD = 0.5      # Fixed threshold for LSH
# PERM_VALUES = [64]   # Using only 64 permutations
# K = 7                # Fixed K = 7
#
# ###############################################
# # Brute Force function
# ###############################################
# def run_brute_force():
#     start_bf = time.time()
#
#     # Load and preprocess data
#     print("Loading and Preprocessing Data (Brute Force)...")
#     train_df = pd.read_csv("train.csv", usecols=[0, 1, 2], sep=",", encoding="utf-8", dtype={"Id": "int32"})
#     test_df = pd.read_csv("test_without_labels.csv", sep=",", encoding="utf-8", dtype={"Id": "int32"})
#
#     # Concatenate Title and Content, remove extra spaces, and convert to lowercase
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
#
#     # Remove the original Title and Content columns
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     # Create shingles using CountVectorizer
#     print("Creating Shingles (Brute Force)...")
#     vectorizer = CountVectorizer(analyzer='char', ngram_range=(SHINGLE_SIZE, SHINGLE_SIZE))
#     train_csr = vectorizer.fit_transform(train_df['FullText'])
#     test_csr = vectorizer.transform(test_df['FullText'])
#
#     # Define an inner function to process documents in chunks
#     def pairwise_jaccard_chunked(test_csr, train_csr, K, train_ids, test_ids, start_test_id=0, global_counter=0):
#         # Convert to binary matrices
#         T = test_csr.astype(bool).astype(int).tocsr()
#         R = train_csr.astype(bool).astype(int).tocsr()
#
#         # Compute intersections between test and training documents
#         intersection = T.dot(R.T)
#         test_sums = T.sum(axis=1).A1  # Number of shingles in each test doc
#         train_sums = R.sum(axis=1).A1  # Number of shingles in each training doc
#
#         local_knn = {}
#
#         # Process each test document in the current chunk
#         for row_i in range(intersection.shape[0]):
#             startptr = intersection.indptr[row_i]
#             endptr = intersection.indptr[row_i + 1]
#             cols = intersection.indices[startptr:endptr]
#             vals = intersection.data[startptr:endptr]
#
#             t_sum = test_sums[row_i]
#             heap = []  # Heap to maintain top K neighbors (using negative distances)
#
#             # Calculate the Jaccard distance for each candidate training document
#             for idx, train_j in enumerate(cols):
#                 union = t_sum + train_sums[train_j] - vals[idx]
#                 dist = 1.0 - (vals[idx] / union) if union > 0 else 1.0
#                 neg_dist = -dist  # Negative distance for max-heap behavior in a min-heap
#                 if len(heap) < K:
#                     heappush(heap, (neg_dist, train_ids[train_j]))
#                 else:
#                     if neg_dist > heap[0][0]:
#                         heapreplace(heap, (neg_dist, train_ids[train_j]))
#
#             # Extract sorted top K neighbors for the test document
#             top_neighbors = [heappop(heap)[1] for _ in range(len(heap))][::-1]
#             test_id = test_ids[start_test_id + row_i]
#             local_knn[test_id] = top_neighbors
#             global_counter += 1
#
#         return local_knn, global_counter
#
#     print("Starting Brute Force Jaccard computation...")
#     knn_results_bf = {}
#     train_ids = train_df['Id'].values
#     test_ids = test_df['Id'].values
#     n_test = test_csr.shape[0]
#
#     global_counter = 0
#     # Process test documents in chunks to manage memory usage
#     for start_idx in range(0, n_test, CHUNK_SIZE):
#         end_idx = min(start_idx + CHUNK_SIZE, n_test)
#         chunk_knn, global_counter = pairwise_jaccard_chunked(
#             test_csr[start_idx:end_idx],
#             train_csr,
#             K,
#             train_ids,
#             test_ids,
#             start_test_id=start_idx,
#             global_counter=global_counter
#         )
#         knn_results_bf.update(chunk_knn)
#
#     print("Finished Brute Force Jaccard computation.")
#     total_time_bf = time.time() - start_bf
#
#     return knn_results_bf, total_time_bf
#
# def run_lsh(num_perm):
#     start_build = time.time()
#
#     print(f"\nLoading and preprocessing data (LSH) with threshold={THRESHOLD}, num_perm={num_perm}...")
#     train_df = pd.read_csv("train.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#     test_df = pd.read_csv("test_without_labels.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
#
#     # Preprocess text: combine Title and Content, clean spaces, and convert to lowercase
#     train_df["FullText"] = (
#             train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     test_df["FullText"] = (
#             test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
#     ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
#
#     # Remove the original Title and Content columns
#     train_df.drop(columns=["Title", "Content"], inplace=True)
#     test_df.drop(columns=["Title", "Content"], inplace=True)
#
#     # Function to generate a set of shingles from the text
#     def get_shingles(text, shingle_size):
#         return {text[i: i + shingle_size] for i in range(len(text) - shingle_size + 1)}
#
#     # Function to create a MinHash signature from a set of shingles
#     def create_minhash(shingles, n_perm):
#         m = MinHash(num_perm=n_perm)
#         for shingle in shingles:
#             m.update(shingle.encode('utf8'))
#         return m
#
#     print("Generating MinHash signatures and shingles (LSH)...")
#     # Compute MinHash and shingle set for training data
#     train_df["MinHash"] = None
#     train_df["Shingles"] = None
#     for idx, row in train_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         train_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#         train_df.at[idx, "Shingles"] = shingle_set
#
#     # Compute MinHash and shingle set for test data
#     test_df["MinHash"] = None
#     test_df["Shingles"] = None
#     for idx, row in test_df.iterrows():
#         shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
#         test_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
#         test_df.at[idx, "Shingles"] = shingle_set
#
#     print("Building LSH index...")
#     # Build LSH index using training documents
#     lsh = MinHashLSH(threshold=THRESHOLD, num_perm=num_perm)
#     for _, row in train_df.iterrows():
#         train_id = str(row["Id"])
#         lsh.insert(train_id, row["MinHash"])
#
#     build_time = time.time() - start_build
#
#     start_query = time.time()
#     print("Finding nearest neighbors with refined brute force Jaccard ranking...")
#     knn_results_lsh = {}
#
#     # Create a dictionary mapping training document IDs to their shingle sets for quick access
#     train_shingles_dict = {str(row["Id"]): row["Shingles"] for _, row in train_df.iterrows()}
#
#     # Function to compute the exact Jaccard similarity between two sets
#     def jaccard_similarity(set1, set2):
#         if not set1 or not set2:
#             return 0.0
#         return len(set1.intersection(set2)) / len(set1.union(set2))
#
#     # Process each test document
#     for idx, row in test_df.iterrows():
#         test_id = row["Id"]
#         test_shingles = row["Shingles"]
#         test_minhash = row["MinHash"]
#         candidate_ids = lsh.query(test_minhash)  # Retrieve candidate neighbors from the same bucket
#         # Remove duplicates if any
#         candidate_ids = list(set(candidate_ids))
#
#         if not candidate_ids:
#             knn_results_lsh[test_id] = []
#             continue
#
#         # Compute the exact Jaccard similarity for each candidate neighbor
#         candidates_similarity = []
#         for cid in candidate_ids:
#             candidate_shingles = train_shingles_dict.get(cid, set())
#             sim = jaccard_similarity(test_shingles, candidate_shingles)
#             candidates_similarity.append((sim, int(cid)))
#
#         # Sort candidates by similarity in descending order and select the top K neighbors
#         candidates_similarity.sort(key=lambda x: x[0], reverse=True)
#         top_neighbors = [cid for sim, cid in candidates_similarity[:K]]
#         knn_results_lsh[test_id] = top_neighbors
#
#     print("Nearest neighbor search completed.")
#     query_time = time.time() - start_query
#
#     return build_time, query_time, knn_results_lsh
#
# ###############################################
# # Function to compute the "fraction"
# ###############################################
# def compute_fraction_of_true_neighbors(brute_force_knn, lsh_knn):
#
#     total_correct = 0
#     total_possible = 0
#
#     for test_id, bf_neighbors in brute_force_knn.items():
#         lsh_neighbors = lsh_knn.get(test_id, [])
#         common = set(bf_neighbors).intersection(set(lsh_neighbors))
#         total_correct += len(common)
#         total_possible += len(bf_neighbors)
#
#     fraction = total_correct / total_possible if total_possible > 0 else 0.0
#     return fraction
#
# ###############################################
# # Execution code
# ###############################################
# print("\n" + "#" * 40)
# print("Running experiment for K = 7")
# print("#" * 40 + "\n")
#
# # Run the brute force method
# bf_knn_results, bf_time = run_brute_force()
# print(f"Brute Force total time: {bf_time:.2f} sec\n")
#
# # Create a table of results
# results_table = []
#
# results_table.append({
#     "Type": "Brute-Force-Jaccard",
#     "BuildTime": 0.0,
#     "QueryTime": bf_time,
#     "TotalTime": bf_time,
#     "fraction of the true K most similar documents that are reported by LSH method as well": "100%",
#     "Parameters": f"K={K}"
# })
#
# # Run the LSH method for each permutation value
# for perm in PERM_VALUES:
#     build_time, query_time, lsh_knn_results = run_lsh(perm)
#     total_time = build_time + query_time
#
#     fraction = compute_fraction_of_true_neighbors(bf_knn_results, lsh_knn_results)
#     fraction_str = f"{fraction*100:.0f}%"
#
#     results_table.append({
#         "Type": "LSH-Jaccard",
#         "BuildTime": round(build_time, 2),
#         "QueryTime": round(query_time, 2),
#         "TotalTime": round(total_time, 2),
#         "fraction of the true K most similar documents that are reported by LSH method as well": fraction_str,
#         "Parameters": f"K={K}, Perm={perm}, threshold={THRESHOLD}"
#     })
#
# # Print the final results table
# df = pd.DataFrame(results_table, columns=[
#     "Type",
#     "BuildTime",
#     "QueryTime",
#     "TotalTime",
#     "fraction of the true K most similar documents that are reported by LSH method as well",
#     "Parameters"
# ])
#
# print("\nFinal Results Table:\n")
# print(df.to_string(index=False))
# print()

###################################################################################################
#Combination of LSH and Brute Force
########################################################################################################

import time
import pandas as pd
import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
from heapq import heappush, heappop, heapreplace

###############################################
# Global parameters
###############################################
SHINGLE_SIZE = 6     # Size of each character shingle
CHUNK_SIZE = 1000    # Chunk size for the brute force computation
THRESHOLD = 0.5      # Fixed threshold for LSH
PERM_VALUES = [64]   # Using only 64 permutations
K = 5               # Fixed K = 5

###############################################
# Brute Force function
###############################################
def run_brute_force():
    start_bf = time.time()

    # Load and preprocess data
    print("Loading and Preprocessing Data (Brute Force)...")
    train_df = pd.read_csv("../AppData/Roaming/JetBrains/PyCharmCE2024.1/scratches/train.csv", usecols=[0, 1, 2], sep=",", encoding="utf-8", dtype={"Id": "int32"})
    test_df = pd.read_csv("../AppData/Roaming/JetBrains/PyCharmCE2024.1/scratches/test_without_labels.csv", sep=",", encoding="utf-8", dtype={"Id": "int32"})

    # Concatenate Title and Content, remove extra spaces, and convert to lowercase
    train_df["FullText"] = (
            train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
    ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

    test_df["FullText"] = (
            test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
    ).str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()

    # Remove the original Title and Content columns
    train_df.drop(columns=["Title", "Content"], inplace=True)
    test_df.drop(columns=["Title", "Content"], inplace=True)

    # Create shingles using CountVectorizer
    print("Creating Shingles (Brute Force)...")
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(SHINGLE_SIZE, SHINGLE_SIZE))
    train_csr = vectorizer.fit_transform(train_df['FullText'])
    test_csr = vectorizer.transform(test_df['FullText'])

    # Define an inner function to process documents in chunks
    def pairwise_jaccard_chunked(test_csr, train_csr, K, train_ids, test_ids, start_test_id=0, global_counter=0):
        # Convert to binary matrices
        T = test_csr.astype(bool).astype(int).tocsr()
        R = train_csr.astype(bool).astype(int).tocsr()

        # Compute intersections between test and training documents
        intersection = T.dot(R.T)
        test_sums = T.sum(axis=1).A1  # Number of shingles in each test doc
        train_sums = R.sum(axis=1).A1  # Number of shingles in each training doc

        local_knn = {}

        # Process each test document in the current chunk
        for row_i in range(intersection.shape[0]):
            startptr = intersection.indptr[row_i]
            endptr = intersection.indptr[row_i + 1]
            cols = intersection.indices[startptr:endptr]
            vals = intersection.data[startptr:endptr]

            t_sum = test_sums[row_i]
            heap = []  # Heap to maintain top K neighbors (using negative distances)

            # Calculate the Jaccard distance for each candidate training document
            for idx, train_j in enumerate(cols):
                union = t_sum + train_sums[train_j] - vals[idx]
                dist = 1.0 - (vals[idx] / union) if union > 0 else 1.0
                neg_dist = -dist  # Negative distance for max-heap behavior in a min-heap
                if len(heap) < K:
                    heappush(heap, (neg_dist, train_ids[train_j]))
                else:
                    if neg_dist > heap[0][0]:
                        heapreplace(heap, (neg_dist, train_ids[train_j]))

            # Extract sorted top K neighbors for the test document
            top_neighbors = [heappop(heap)[1] for _ in range(len(heap))][::-1]
            test_id = test_ids[start_test_id + row_i]
            local_knn[test_id] = top_neighbors
            global_counter += 1

        return local_knn, global_counter

    print("Starting Brute Force Jaccard computation...")
    knn_results_bf = {}
    train_ids = train_df['Id'].values
    test_ids = test_df['Id'].values
    n_test = test_csr.shape[0]

    global_counter = 0
    # Process test documents in chunks to manage memory usage
    for start_idx in range(0, n_test, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, n_test)
        chunk_knn, global_counter = pairwise_jaccard_chunked(
            test_csr[start_idx:end_idx],
            train_csr,
            K,
            train_ids,
            test_ids,
            start_test_id=start_idx,
            global_counter=global_counter
        )
        knn_results_bf.update(chunk_knn)

    print("Finished Brute Force Jaccard computation.")
    total_time_bf = time.time() - start_bf

    return knn_results_bf, total_time_bf

def run_lsh(num_perm):
    start_build = time.time()

    print(f"\nLoading and preprocessing data (LSH) with threshold={THRESHOLD}, num_perm={num_perm}...")
    train_df = pd.read_csv("../AppData/Roaming/JetBrains/PyCharmCE2024.1/scratches/train.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")
    test_df = pd.read_csv("../AppData/Roaming/JetBrains/PyCharmCE2024.1/scratches/test_without_labels.csv", usecols=["Id", "Title", "Content"], sep=",", encoding="utf-8")

    # Preprocess text: combine Title and Content, clean spaces, and convert to lowercase
    train_df["FullText"] = (
            train_df["Title"].fillna("") + " " + train_df["Content"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

    test_df["FullText"] = (
            test_df["Title"].fillna("") + " " + test_df["Content"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()

    # Remove the original Title and Content columns
    train_df.drop(columns=["Title", "Content"], inplace=True)
    test_df.drop(columns=["Title", "Content"], inplace=True)

    # Function to generate a set of shingles from the text
    def get_shingles(text, shingle_size):
        return {text[i: i + shingle_size] for i in range(len(text) - shingle_size + 1)}

    # Function to create a MinHash signature from a set of shingles
    def create_minhash(shingles, n_perm):
        m = MinHash(num_perm=n_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf8'))
        return m

    print("Generating MinHash signatures and shingles (LSH)...")
    # Compute MinHash and shingle set for training data
    train_df["MinHash"] = None
    train_df["Shingles"] = None
    for idx, row in train_df.iterrows():
        shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
        train_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
        train_df.at[idx, "Shingles"] = shingle_set

    # Compute MinHash and shingle set for test data
    test_df["MinHash"] = None
    test_df["Shingles"] = None
    for idx, row in test_df.iterrows():
        shingle_set = get_shingles(row["FullText"], SHINGLE_SIZE)
        test_df.at[idx, "MinHash"] = create_minhash(shingle_set, num_perm)
        test_df.at[idx, "Shingles"] = shingle_set

    print("Building LSH index...")
    # Build LSH index using training documents
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=num_perm)
    for _, row in train_df.iterrows():
        train_id = str(row["Id"])
        lsh.insert(train_id, row["MinHash"])

    build_time = time.time() - start_build

    start_query = time.time()
    print("Finding nearest neighbors with refined brute force Jaccard ranking...")
    knn_results_lsh = {}

    # Create a dictionary mapping training document IDs to their shingle sets for quick access
    train_shingles_dict = {str(row["Id"]): row["Shingles"] for _, row in train_df.iterrows()}

    # Function to compute the exact Jaccard similarity between two sets
    def jaccard_similarity(set1, set2):
        if not set1 or not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))

    # Process each test document
    for idx, row in test_df.iterrows():
        test_id = row["Id"]
        test_shingles = row["Shingles"]
        test_minhash = row["MinHash"]
        candidate_ids = lsh.query(test_minhash)  # Retrieve candidate neighbors from the same bucket
        # Remove duplicates if any
        candidate_ids = list(set(candidate_ids))

        if not candidate_ids:
            knn_results_lsh[test_id] = []
            continue

        # Compute the exact Jaccard similarity for each candidate neighbor
        candidates_similarity = []
        for cid in candidate_ids:
            candidate_shingles = train_shingles_dict.get(cid, set())
            sim = jaccard_similarity(test_shingles, candidate_shingles)
            candidates_similarity.append((sim, int(cid)))

        # Sort candidates by similarity in descending order and select the top K neighbors
        candidates_similarity.sort(key=lambda x: x[0], reverse=True)
        top_neighbors = [cid for sim, cid in candidates_similarity[:K]]
        knn_results_lsh[test_id] = top_neighbors

    print("Nearest neighbor search completed.")
    query_time = time.time() - start_query

    return build_time, query_time, knn_results_lsh

###############################################
# Function to compute the "fraction"
###############################################
def compute_fraction_of_true_neighbors(brute_force_knn, lsh_knn):

    total_correct = 0
    total_possible = 0

    for test_id, bf_neighbors in brute_force_knn.items():
        lsh_neighbors = lsh_knn.get(test_id, [])
        common = set(bf_neighbors).intersection(set(lsh_neighbors))
        total_correct += len(common)
        total_possible += len(bf_neighbors)

    fraction = total_correct / total_possible if total_possible > 0 else 0.0
    return fraction

###############################################
# Execution code
###############################################
print("\n" + "#" * 40)
print("Running experiment for K = 5")
print("#" * 40 + "\n")

# Run the brute force method
bf_knn_results, bf_time = run_brute_force()
print(f"Brute Force total time: {bf_time:.2f} sec\n")

# Create a table of results
results_table = []

results_table.append({
    "Type": "Brute-Force-Jaccard",
    "BuildTime": 0.0,
    "QueryTime": bf_time,
    "TotalTime": bf_time,
    "fraction of the true K most similar documents that are reported by LSH method as well": "100%",
    "Parameters": f"K={K}"
})

# Run the LSH method for each permutation value
for perm in PERM_VALUES:
    build_time, query_time, lsh_knn_results = run_lsh(perm)
    total_time = build_time + query_time

    fraction = compute_fraction_of_true_neighbors(bf_knn_results, lsh_knn_results)
    fraction_str = f"{fraction*100:.0f}%"

    results_table.append({
        "Type": "LSH-Jaccard",
        "BuildTime": round(build_time, 2),
        "QueryTime": round(query_time, 2),
        "TotalTime": round(total_time, 2),
        "fraction of the true K most similar documents that are reported by LSH method as well": fraction_str,
        "Parameters": f"K={K}, Perm={perm}, threshold={THRESHOLD}"
    })

# Print the final results table
df = pd.DataFrame(results_table, columns=[
    "Type",
    "BuildTime",
    "QueryTime",
    "TotalTime",
    "fraction of the true K most similar documents that are reported by LSH method as well",
    "Parameters"
])

print("\nFinal Results Table:\n")
print(df.to_string(index=False))
print()