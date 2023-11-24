import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image and convert it to grayscale
image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Optionally resize the image
image = cv2.resize(image, (500, 700))  # Uncomment this line for resizing

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(image, full_matrices=False)

def low_rank_approximation(A, k):
    Ak = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return Ak

# Vary k and plot the resultant k-rank approximation
num_values = 12
k_values = np.linspace(1, min(image.shape)//15, num=num_values, dtype=int)
# print(k_values)

plt.figure(figsize=(14, 6))
for i, k in enumerate(k_values):
    approx_image = low_rank_approximation(image, k)
    plt.subplot(2, 6, i+1)
    plt.imshow(approx_image, cmap='gray')
    plt.title(f'k = {k}')
    plt.axis('off')

plt.tight_layout()
plt.show()
