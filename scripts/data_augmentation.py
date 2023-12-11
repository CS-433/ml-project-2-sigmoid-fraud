import os
import cv2 as cv
import numpy as np
from skimage import color  # Import color from skimage
import albumentations as A

def parse_mask_to_binary(input_mask, grayscale_threshold=0.5):
    """
    Function that takes an RGB mask and outputs the corresponding binary mask
    """
    binary_mask = np.where(color.rgb2gray(input_mask) <= grayscale_threshold, 0, 1)
    return binary_mask

TRAINING_DIR = "/content/gdrive/MyDrive/ml-project-2-sigmoid-fraud-main/data/training"

# Initialize an empty list to store training filenames
training_filenames = []

# Print the contents of the images directory
image_dir = os.path.join(TRAINING_DIR, 'images')
print("Contents of the images directory:", os.listdir(image_dir))

# Iterate over the directory tree of training images and store sorted filenames
for root, dirs, files in sorted(os.walk(image_dir)):
    training_filenames.extend(sorted(files))

# Check if any filenames are found
if not training_filenames:
    print("No training filenames found. Check directory path and contents.")
else:
    # Initialize empty lists to store training images and their labels
    training_images, training_image_labels = ([], [])

    # Loop through each filename and process the images and labels
    for fdx, file in enumerate(training_filenames):
        img_path = os.path.join(TRAINING_DIR, 'images', file)
        groundtruth_mask_path = os.path.join(TRAINING_DIR, 'groundtruth', file)
        
        # Read the image using OpenCV
        img = cv.imread(img_path)

        # Read the ground truth mask
        groundtruth_mask = cv.imread(groundtruth_mask_path)
        
        # Check if the images were successfully loaded
        if img is None or groundtruth_mask is None:
            print(f"Error loading image or mask: {file}")
            continue

        # Convert the ground truth mask to binary format
        groundtruth_mask_binary = parse_mask_to_binary(groundtruth_mask)[:, :, None]

        training_images.append(img)
        training_image_labels.append(groundtruth_mask_binary)

        if fdx % 20 == 0:
            print('img number:', fdx)

    # Convert lists to NumPy arrays
    training_images = np.array(training_images)
    training_image_labels = np.array(training_image_labels)

    # Save the training images and labels
    np.save('/content/gdrive/MyDrive/ml-project-2-sigmoid-fraud-main/data/derivatives/training_imgs.npy', training_images)
    np.save('/content/gdrive/MyDrive/ml-project-2-sigmoid-fraud-main/data/derivatives/training_imgs_labels.npy', training_image_labels)

    # Print the shapes of the dataset and labels
    print("dataset shape:", training_images.shape, "labels shape:", training_image_labels.shape)


training_imgs = training_images 
training_imgs_labels = training_image_labels


def augment_data(images, masks, num_augmentations=3):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        # Add more augmentations as needed
    ])

    augmented_images = []
    augmented_masks = []

    # Iterate over each image and mask pair
    for idx in range(len(images)):
        for _ in range(num_augmentations):
            augmented = transform(image=images[idx], mask=masks[idx])
            augmented_images.append(augmented['image'])
            augmented_masks.append(augmented['mask'])

    # Convert lists to NumPy arrays
    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)

    return augmented_images, augmented_masks

num_augmentations = 3  # Adjust the number of augmentations as needed

# Perform data augmentation
augmented_imgs, augmented_labels = augment_data(training_imgs, training_imgs_labels, num_augmentations=num_augmentations)

# Concatenate the augmented dataset with the initial dataset
augmented_dataset = np.concatenate([training_imgs, augmented_imgs], axis=0)
augmented_labels_dataset = np.concatenate([training_imgs_labels, augmented_labels], axis=0)

# Print the shapes of the combined dataset and labels
print("Combined dataset shape:", augmented_dataset.shape, "Combined labels shape:", augmented_labels_dataset.shape)


