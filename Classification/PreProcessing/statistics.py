import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from shutil import copyfile

# Function to count files in a directory or a list of files
def count_files(directory_or_files):
    if isinstance(directory_or_files, str):  # If a single directory path is provided
        return len([f for f in os.listdir(directory_or_files) if os.path.isfile(os.path.join(directory_or_files, f))])
    elif isinstance(directory_or_files, list):  # If a list of files is provided
        return len(directory_or_files)
    else:
        raise ValueError("Invalid input type. Provide either a directory path or a list of files.")


# Function to compute color histogram as features for an image
def compute_histogram(image_path):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to compute cosine similarity matrix for a set of feature vectors
def compute_similarity_matrix(feature_matrix):
    return cosine_similarity(feature_matrix)

# Function to print statistics for a given class
def print_class_stats(class_name, class_path):
    num_files = count_files(class_path)
    print(f"{class_name}: {num_files}")

# Function to print total statistics
def print_total_stats():
    total_fire_files = count_files(Training_Fire_path) + count_files(Testing_Fire_path)
    total_no_fire_files = count_files(Training_No_Fire_path) + count_files(Testing_No_Fire_path)
    total_files = total_fire_files + total_no_fire_files

    print("Total Statistics Initial Dataset:")
    print(f"Total number of images: {total_files}")
    print(f"Number of Fire images: {total_fire_files}")
    print(f"Number of No Fire images: {total_no_fire_files}")
    print(f"Percentage of Fire Training Images: { count_files(Training_Fire_path) / (count_files(Training_Fire_path)+(count_files(Training_No_Fire_path))) * 100:.2f}%")
    print(f"Percentage of No Fire Training Images: { count_files(Training_No_Fire_path) / (count_files(Training_Fire_path)+(count_files(Training_No_Fire_path))) * 100:.2f}%")
    print(f"Percentage of Fire Test Images: { count_files(Testing_Fire_path) / (count_files(Testing_Fire_path)+(count_files(Testing_No_Fire_path))) * 100:.2f}%")
    print(f"Percentage of No Fire Test Images: { count_files(Testing_No_Fire_path) / (count_files(Testing_Fire_path)+(count_files(Testing_No_Fire_path))) * 100:.2f}%")
    print(f"Percentage of Fire images: {(total_fire_files / total_files) * 100:.2f}%")
    print(f"Percentage of No Fire images: {(total_no_fire_files / total_files) * 100:.2f}%")

# Get the current folder paths of the dataset
Training_Fire_path = 'Dataset/Classification/Training/Fire'
Training_No_Fire_path = 'Dataset/Classification/Training/No_Fire'

Testing_Fire_path = 'Dataset/Classification/Test/Fire'
Testing_No_Fire_path = 'Dataset/Classification/Test/No_Fire'

# Print total statistics
print_total_stats()

# Compute features and diversity scores for each image
fire_images = [os.path.join(Training_Fire_path, f) for f in os.listdir(Training_Fire_path)]
no_fire_images = [os.path.join(Training_No_Fire_path, f) for f in os.listdir(Training_No_Fire_path)]


# Extract features (color histograms) for each image
fire_feature_matrix = np.array([compute_histogram(image) for image in fire_images])
no_fire_feature_matrix = np.array([compute_histogram(image) for image in no_fire_images])

# Compute cosine similarity matrix
fire_similarity_matrix = compute_similarity_matrix(fire_feature_matrix)
no_fire_similarity_matrix = compute_similarity_matrix(no_fire_feature_matrix)

# Define diversity metric (average dissimilarity)
fire_diversity_scores = np.mean(1 - fire_similarity_matrix, axis=1)
no_fire_diversity_scores = np.mean(1 - no_fire_similarity_matrix, axis=1)

# Sort images based on diversity scores
fire_sorted_indices = np.argsort(fire_diversity_scores)
no_fire_sorted_indices = np.argsort(no_fire_diversity_scores)

# Choose a subset (e.g., top 1000 diverse images)
subset_size = 1000
fire_selected_subset_indices = fire_sorted_indices[:subset_size]
no_fire_selected_subset_indices = no_fire_sorted_indices[:subset_size]

# Output folders for the new dataset
New_Dataset_Fire_path = 'New_Dataset/Training/Fire'
New_Dataset_No_Fire_path = 'New_Dataset/Training/No_Fire'

# Create output folders if they don't exist
os.makedirs(New_Dataset_Fire_path, exist_ok=True)
os.makedirs(New_Dataset_No_Fire_path, exist_ok=True)

# Copy the selected diverse subset to the new dataset folders
for i in fire_selected_subset_indices:
    image_path = fire_images[i]
    destination_folder = New_Dataset_Fire_path

    # Copy the image to the destination folder
    destination_path = os.path.join(destination_folder, os.path.basename(image_path))
    copyfile(image_path, destination_path)

# Copy the selected diverse subset to the new dataset folders
for i in no_fire_selected_subset_indices:
    image_path = no_fire_images[i]
    destination_folder = New_Dataset_No_Fire_path
    
    # Copy the image to the destination folder
    destination_path = os.path.join(destination_folder, os.path.basename(image_path))
    copyfile(image_path, destination_path)

# Print total statistics for the new dataset
print("Total Statistics for New Dataset:")
print_class_stats("New Dataset Fire", New_Dataset_Fire_path)
print_class_stats("New Dataset No Fire", New_Dataset_No_Fire_path)
