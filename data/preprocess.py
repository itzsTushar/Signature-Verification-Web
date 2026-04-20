
from PIL import Image, ImageOps
import numpy as np
from numpy import asarray, pad
from math import floor, ceil
import os
# ---------------- PREPROCESS FUNCTION ----------------
def process_image(im, img_dim=(224, 224)):
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
