# imports for downloading files
import urllib.request
import zipfile
import os

"""Download the Dataset for Classification:"""

# Get files in current working directory
files = os.listdir()

if 'Dataset' and 'Dataset.zip' not in files:
    # This url points to the download of the .zip file for the classification training
    url = 'https://www.dropbox.com/scl/fi/9sxb3s88hw2zr2f0bbvf9/Dataset.zip?rlkey=8s4bobjz0b7ee68vjt384cjk1&dl=1'

    # Download the zip file
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    # Specify the local filename for the downloaded zip file
    zip_filename = 'Dataset.zip'

    with open(zip_filename, 'wb') as f:
        f.write(data)

    # Unzip the downloaded file
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        # Extract all contents to the current working directory
        zip_ref.extractall()
