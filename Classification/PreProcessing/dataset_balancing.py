import os

import cv2

# Check how many pictures are in each folder
def print_class_stats(class_name, path):
    print(class_name)
    print("Number of images:", len(os.listdir(path)))
    
print_class_stats("Training Fire", "Dataset/Classification/Training/Fire")
print_class_stats("Training No Fire", "Dataset/Classification/Training/No_Fire")

# Take pictures in No_fire folder and transpose them and add them to the No_fire folder
# This will double the amount of pictures in the No_fire folder

# Get the current folder paths of the dataset
Training_Fire_path = 'Dataset/Classification/Training/Fire'
Training_No_Fire_path = 'Dataset/Classification/Training/No_Fire'

# Get the list of images in the No_fire folder
fire_images = [os.path.join(Training_Fire_path, f) for f in os.listdir(Training_Fire_path)]
no_fire_images = [os.path.join(Training_No_Fire_path, f) for f in os.listdir(Training_No_Fire_path)]

i = len(no_fire_images)
fire_images_length = len(fire_images)
# Transpose the images and add them to the No_fire folder
for image in no_fire_images:
    if i == fire_images_length:
        break
    img = cv2.imread(image)
    img = cv2.transpose(img)
    cv2.imwrite('Dataset/Classification/Training/No_Fire/transposed_' + str(i) + '.jpg', img)
    i += 1

print_class_stats("Training Fire", "Dataset/Classification/Training/Fire")
print_class_stats("Training No Fire", "Dataset/Classification/Training/No_Fire")