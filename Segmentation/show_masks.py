import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

cwd = os.getcwd()
folder_path = cwd+"/Segmentation/Masks"
# Set the folder_path as the working directory
os.chdir(folder_path)

# Get the list of files in the folder
file_list = os.listdir()

# Print the list of files
print(len(file_list))

mask = cv2.imread(file_list[1], cv2.IMREAD_UNCHANGED)

print(mask.shape)
# Multiply the matrix by 255
mask = mask * 255

# Open the image and show it
plt.imshow(mask)
plt.show()

