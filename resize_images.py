import glob
import os
from torchvision.transforms import transforms
import cv2


image_path = 'data/images_full'
masks_path = 'data/masks_full'
img_dim = (512, 512)

masks_file_list = glob.glob(os.path.join(masks_path, '*.png'))
new_masks_path = 'data/masks_resized'

for i in range(0, len(masks_file_list)):
    mask = cv2.imread(masks_file_list[i], cv2.IMREAD_UNCHANGED)
    mask = mask*255
    mask = cv2.resize(mask, img_dim)

    # save resized images to new path
    cv2.imwrite(os.path.join(new_masks_path, os.path.basename(masks_file_list[i])), mask)