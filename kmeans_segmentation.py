import cv2
import numpy as np
from sklearn.cluster import KMeans

def kmeans_segmentation(image):
    image_2d = image.reshape((-1, 1))  
    image_2d = np.float32(image_2d)

    kmeans = KMeans(n_clusters=2, random_state=42)  
    kmeans.fit(image_2d)
    segmented_img = kmeans.labels_.reshape(image.shape)

    result_img = np.zeros_like(image)
    result_img[segmented_img == 1] = 255  
    result_img[segmented_img == 0] = 0   

    return result_img
