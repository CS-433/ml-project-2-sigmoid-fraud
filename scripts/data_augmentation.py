import albumentations as A


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

DEVICE = "cuda" 
DEVICE

unet_config = {
    "model": "U-Net",
    "patch_size": 16,
    "n_epochs": 50,
    "batch_size": 10,
    "optimizer": "AdamW",
    "loss_function": "BCELoss",
    "learning_rate": 1e-4,
    "prediction_transform": "identity_detach",
    "labels_transform": "unet_reshape",
    "scheduler": "CosineAnnealing",
    "device": DEVICE,
    "test_size": 0.1
}

unet = train_model(X=augmented_dataset, y=augmented_labels_dataset, config=unet_config)


img_idx = 4
im = cv.imread(f'/content/drive/MyDrive/ml-project-2-sigmoid-fraud-main/data/training/images/satImage_00{img_idx}.png')
im_gt = cv.imread(f'/content/drive/MyDrive/ml-project-2-sigmoid-fraud-main/data/training/groundtruth/satImage_00{img_idx}.png')
im_gt_gs = np.where(color.rgb2gray(im_gt) > .5, 1, 0)


visualize_predicition_unet(img=im, gt=im_gt_gs, model=unet)

def test_masks(test_directory, model, visualize_number=6):
  


  test_filepaths = []
  
  for root, dirs, files in sorted(os.walk(test_directory))[1:]: test_filepaths.append(root+'/'+files[0])
  

  output_masks = [] #Here we go baby
  output_masks_filenames = []
  for fdx, file in enumerate(sorted(test_filepaths)):

    test_img = cv.imread(file)
    test_img = torch.tensor(test_img[:,:,:,None].T, dtype=torch.float32).to(device=DEVICE)

    mask = np.where(identity_detach(model(test_img)) > 0.5, 1, 0)[0][0]

    print(f"data/test_masks/{file.split('/')[-1]}")
    output_filename = f"data/test_masks/{file.split('/')[-1]}"
    output_masks_filenames.append(output_filename)
    cv.imwrite(output_filename, 255*mask.T)

    output_masks.append(mask.T)

  test_ids = np.random.choice(len(output_masks), size = visualize_number)
  fig, ax = plt.subplots(visualize_number, 2, figsize = (visualize_number*3, visualize_number*3))

  for i in range(visualize_number):
    test_img = cv.imread(test_filepaths[test_ids[i]])
    ax[i, 0].imshow(test_img)
    ax[i, 1].imshow(output_masks[test_ids[i]])

  plt.show()
  return output_masks, output_masks_filenames

x_test_dir = '/content/drive/MyDrive/ml-project-2-sigmoid-fraud-main/data/test_set_images'
predicted_test_masks, mask_filenames = test_masks(x_test_dir, model = unet,visualize_number=10)

