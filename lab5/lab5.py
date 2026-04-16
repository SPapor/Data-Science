import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_image_colors(image_path, k=10):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[-] Помилка: не вдалося знайти або завантажити зображення '{image_path}'")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    pixels = blurred_image.reshape((-1, 3))

    print(f"[+] Обробка {image_path} (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    dominant_colors = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_

    clustered_pixels = dominant_colors[labels]
    clustered_image = clustered_pixels.reshape(image.shape)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'Оригінал: {image_path}')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'Кластеризація (k={k})')
    plt.imshow(clustered_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    images_to_process = [
        'img.png',
        'img_1.png',
        'img_2.png',
        'img_3.png',
        'img_4.png'
    ]

    CLUSTERS_COUNT = 10

    for img_path in images_to_process:
        cluster_image_colors(img_path, k=CLUSTERS_COUNT)