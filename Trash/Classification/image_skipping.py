import os
import shutil
import shutil

skip = 10 #images to skip

# Specify the folder name
current_folder = "current_folder"
new_folder = "new_folder"
# Function to recursively process the files in a directory




def process_files(current_folder, new_folder):
    # Get the list of files in the current folder
    files = os.listdir(current_folder)
    # Check if new_folder exists, if not create it
    print('new folder: ',new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    # Retrieve every 30th element from the list
    filtered_files = list(filter(lambda x: os.path.isfile(os.path.join(current_folder, x)) and files.index(x) % skip == 0, files))
    if len(filtered_files) > 0:
        print('Folder:' + current_folder)
        print('Files not processed size:', len(files))
        print('Files processed size:', len(filtered_files))
        # Process the filtered files (e.g., move them to a new folder)
        for file in filtered_files:
            source_path = os.path.join(current_folder, file)
            destination_path = os.path.join(new_folder, file)
            shutil.move(source_path, destination_path)

    # Recursively process subdirectories
    for root, dirs, files in os.walk(current_folder):
        for dir in dirs:
            process_files(os.path.join(root, dir), os.path.join(new_folder, dir))

# Start processing from the current folder
os.makedirs(new_folder)
process_files(current_folder, new_folder)
