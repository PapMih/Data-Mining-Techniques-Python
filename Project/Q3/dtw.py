import numpy as np
import pandas as pd
import ast
import time

# Load the dataset
file_path = "dtw_test.csv"
df = pd.read_csv(file_path)

# DTW function using dynamic programming
def dtw_distance(series_a, series_b):
    n, m = len(series_a), len(series_b)

    # Initialize the DTW matrix with infinity (except for the (0,0) position which is set as 0)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)  # n+1 ensures correct boundary handling
    dtw_matrix[0, 0] = 0

    # Compute distances using absolute difference
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(series_a[i - 1] - series_b[j - 1])  # Absolute difference
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],    # Insertion
                dtw_matrix[i, j - 1],    # Deletion
                dtw_matrix[i - 1, j - 1] # Match
            )

    return dtw_matrix[n, m]  # The final DTW distance

# Function to convert string representation of lists into numpy arrays
def convert_series(series):
    if isinstance(series, str):
        series = ast.literal_eval(series)  # Convert string to list
    return np.array(series, dtype=float).flatten()  # Ensure 1D array

# Convert the series columns to numpy arrays
df["series_a"] = df["series_a"].apply(convert_series)
df["series_b"] = df["series_b"].apply(convert_series)

# Compute DTW distances
dtw_distances = []
execution_times = []  # Store time for each computation

total_start_time = time.time()  # Start measuring total execution time

for i, row in df.iterrows():
    series_a = row["series_a"]
    series_b = row["series_b"]

    start_time = time.time()  # Start time for each DTW calculation
    distance = dtw_distance(series_a, series_b)
    end_time = time.time()  # End time

    elapsed_time = end_time - start_time  # Compute time taken
    execution_times.append(elapsed_time)  # Store it
    dtw_distances.append(distance)  # Store DTW distance

    print(f"Computed DTW for ID={row['id']} in {elapsed_time:.4f} seconds")

total_end_time = time.time()  # Stop measuring total execution time
total_execution_time = total_end_time - total_start_time

# Add distances to DataFrame
df["DTW distance"] = dtw_distances

# Save results to CSV
output_path = "dtw_results.csv"
df[["id", "DTW distance"]].to_csv(output_path, index=False)

print(f"Results saved to {output_path}")
print(f"Total execution time: {total_execution_time:.4f} seconds")
print(f"Average time per DTW calculation: {np.mean(execution_times):.4f} seconds")
