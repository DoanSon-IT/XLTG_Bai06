import numpy as np
import skfuzzy as fuzz
import cv2

# Hàm phân cụm FCM
def fcm_segmentation(image):
    image_2d = image.reshape((-1, 1)) 
    image_2d = np.float32(image_2d)

    
    n_clusters = 3
    cntr, u, _, _, _, _, _ = fuzz.cmeans(image_2d.T, n_clusters, 2, error=0.005, maxiter=1000)
    
    labels = np.argmax(u, axis=0)
    segmented_img = labels.reshape(image.shape)

    result_img = np.zeros_like(image)
    result_img[segmented_img == 1] = 255 
    result_img[segmented_img == 0] = 0   

    return result_img
