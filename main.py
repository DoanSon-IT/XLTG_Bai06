import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from fcm_segmentation import fcm_segmentation

def display_result(window_name, image):
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(window_name)
    plt.axis('off') 
    plt.show()

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    
    return enhanced_image

def kmeans_segmentation(image, n_clusters=2):
    image_2d = image.reshape((-1, 1)) 
    image_2d = np.float32(image_2d)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_2d)

    labels = kmeans.labels_
    segmented_img = labels.reshape(image.shape)

    result_img = np.zeros_like(image)
    result_img[segmented_img == 1] = 255 
    result_img[segmented_img == 0] = 0  

    return result_img

def save_image(image, image_name, prefix="output"):
    output_folder = './output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.basename(image_name)
    name, ext = os.path.splitext(base_name)

    filename = os.path.join(output_folder, f"{prefix}_{name}{ext}")

    cv2.imwrite(filename, image)
    print(f"Đã lưu ảnh tại: {filename}")

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc được ảnh từ {image_path}")
        return

    processed_image = preprocess_image(image)
    
    kmeans_result = kmeans_segmentation(processed_image)
    display_result("KMeans Segmentation", kmeans_result)
    save_image(kmeans_result, image_path, prefix="K_means")
    
    fcm_result = fcm_segmentation(processed_image)
    display_result("FCM Segmentation", fcm_result)
    save_image(fcm_result, image_path, prefix="FCM") 


if __name__ == "__main__":
    image_path = './AnhVeTinh_imgs/anh_ve_tinh_3.png'
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc được ảnh từ {image_path}")
    else:
        main(image_path)
