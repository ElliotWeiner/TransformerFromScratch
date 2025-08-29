
import os
import numpy as np
from PIL import Image, ImageFilter

def load_image_dataset(base_dir, image_size=(128, 128), train_split=0.8, batch_size=32):
    """
    Load images from subfolders as numpy arrays with one-hot labels, split into batches.

    Args:
        base_dir (str): Path to base folder containing subfolders for each class.
        image_size (tuple): Resize images to this size (H, W).
        train_split (float): Fraction of data to use for training.
        batch_size (int): Number of samples per batch.

    Returns:
        train_data, test_data: lists of dicts {'inputs':..., 'labels':...} per batch.
    """
    X = []
    y = []

    class_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    num_classes = len(class_folders)

    for label_idx, class_name in enumerate(class_folders):
        class_dir = os.path.join(base_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, fname)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size[:2])
                img_array = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
                X.append(img_array)

                # one-hot label
                one_hot = np.zeros(num_classes, dtype=np.float32)
                one_hot[label_idx] = 1.0
                y.append(one_hot)

    X = np.stack(X)
    y = np.stack(y)


    # shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # split train/test
    split_idx = int(len(X) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # function to batch data
    def make_batches(X, y, batch_size):
        batches = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            batches.append({'inputs': np.array(batch_X), 'labels': np.array(batch_y)})
        return batches

    train_data = make_batches(X_train, y_train, batch_size)
    test_data = make_batches(X_test, y_test, batch_size)

    return train_data, test_data


import numpy as np
from PIL import Image, ImageEnhance
import random

def augment_batch(batch, rotation_range=15, brightness_range=(0.8, 1.2),
                  contrast_range=(0.8, 1.2), color_range=(0.8, 1.2)):
    """
    Apply random rotation and color jitter to a batch of images.

    Args:
        batch (dict): {'inputs': np.array(B,H,W,C), 'labels': np.array(B,C)}
        rotation_range (float): max absolute rotation in degrees
        brightness_range (tuple): min/max factor for brightness
        contrast_range (tuple): min/max factor for contrast
        color_range (tuple): min/max factor for saturation (color)
    
    Returns:
        Augmented batch (same dict structure, inputs modified)
    """
    B, H, W, C = batch['inputs'].shape
    augmented_images = []

    for i in range(B):
        img = batch['inputs'][i]
        # Convert to PIL Image
        img_pil = Image.fromarray((img * 255).astype(np.uint8))

        # Random rotation
        angle = random.uniform(-rotation_range, rotation_range)
        img_pil = img_pil.rotate(angle)

        # Random flip
        if random.random() > 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.3:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.5)))

        # Random color jitter
        img_pil = ImageEnhance.Brightness(img_pil).enhance(random.uniform(*brightness_range))
        img_pil = ImageEnhance.Contrast(img_pil).enhance(random.uniform(*contrast_range))
        img_pil = ImageEnhance.Color(img_pil).enhance(random.uniform(*color_range))

        # Convert back to NumPy array in [0,1]
        img_aug = np.array(img_pil, dtype=np.float32) / 255.0
        augmented_images.append(img_aug)

    batch_aug = batch.copy()
    batch_aug['inputs'] = np.stack(augmented_images)
    return batch_aug
