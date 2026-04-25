from PIL import Image, ImageOps
import numpy as np
from numpy import asarray, pad
from math import floor, ceil
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
target_stats = {
 'height': 350,
 'width': 544,
 'aspect_ratio': 1.62,
 'channels': 3,
 'mean_intensity': 242.16,
 'std_intensity': 20.34,
 'blur_score': 392.8,
 'foreground_ratio': 0.014
}

"""def preprocess_to_dataset_style(img, img_name, target_stats=target_stats):
    if img is None:
        raise ValueError("Image not found")

    original = img.copy()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Normalize intensity (match mean & std)
    current_mean = np.mean(gray)
    current_std = np.std(gray)

    target_mean = target_stats['mean_intensity']
    target_std = target_stats['std_intensity']

    if current_std < 1e-5:
        current_std = 1.0

    normalized = (gray - current_mean) / current_std
    normalized = normalized * target_std + target_mean
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Step 3: Threshold
    _, thresh = cv2.threshold(
        normalized, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 4: Crop
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = thresh[y:y+h, x:x+w]
    else:
        cropped = thresh

    # Step 5: Resize
    target_size = (int(target_stats['width']), int(target_stats['height']))
    resized = cv2.resize(cropped, target_size)

    # Step 6: Convert to 3 channel
    final = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    final = 255 - final

    print(final.shape)

    #  SAVE IMAGE
    save_dir = "uploads/reference"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, final)

    #  RETURN PATH (as requested)
    return save_path """
# ---------------- PREPROCESS FUNCTION ----------------
def process_image(im, img_dim=(224, 224)):
    print(im.size)
    w_new, h_new = img_dim

    im = im.convert('L')
    im = ImageOps.invert(im)

    w, h = im.size
    w_prime = min(w_new, w * h_new // h)
    h_prime = h * w_prime // w

    im = im.resize((w_prime, h_prime))
    img = asarray(im)

    top = floor((h_new - h_prime) / 2)
    bottom = ceil((h_new - h_prime) / 2)
    left = floor((w_new - w_prime) / 2)
    right = ceil((w_new - w_prime) / 2)

    img = pad(img, ((top, bottom), (left, right)))

    img = img.reshape(224, 224, 1)   #SAME AS TRAINING

    return np.expand_dims(img, axis=0)

