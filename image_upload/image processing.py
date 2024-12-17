import cv2
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

# Load the image from file
#image_path = "image.jpg"  # Replace with the actual path of your image
image_path = r"C:\Users\Rishi\OneDrive\Desktop\DiseaseDetection\plant disease detection\image_upload\image.jpg"
#image = cv2.imread(image_path)
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found or unable to load.")

# Convert from BGR to RGB (OpenCV reads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize the image (scaling pixel values to 0-255 range)
normalized_image = cv2.normalize(image_rgb, None, 0, 255, cv2.NORM_MINMAX)

# Apply horizontal flip to the image
horizontal_flip = cv2.flip(image_rgb, 1)

# Rotate the image by 45 degrees
height, width = image_rgb.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (width, height))

# Adjust the brightness of the image
pil_image = Image.fromarray(image_rgb)
brightness_enhancer = ImageEnhance.Brightness(pil_image)
bright_image = np.array(brightness_enhancer.enhance(1.5))  # Increase brightness by 1.5 times

# Enhance the contrast of the image
contrast_enhancer = ImageEnhance.Contrast(pil_image)
contrast_image = np.array(contrast_enhancer.enhance(1.5))  # Increase contrast by 1.5 times

# Zoom into the image (crop the center 70% of the original image)
zoom_factor = 0.7
new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
top, left = (height - new_height) // 2, (width - new_width) // 2
zoomed_image = image_rgb[top:top + new_height, left:left + new_width]

# Adjust the saturation of the image
saturation_enhancer = ImageEnhance.Color(pil_image)
saturated_image = np.array(saturation_enhancer.enhance(1.5))  # Increase saturation by 1.5 times

# Adjust the hue of the image (change the hue channel in HSV space)
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
hsv_image[:, :, 0] = (hsv_image[:, :, 0] + 30) % 180  # Add 30 to hue channel
hue_adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Apply edge detection using the Canny algorithm
edges_image = cv2.Canny(image_rgb, 100, 200)

# Apply histogram equalization (for color images in HSV space)
hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
histogram_equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(image_rgb, (5, 5), 0)

# Remove noise using Non-Local Means Denoising
denoised_image = cv2.fastNlMeansDenoisingColored(image_rgb, None, 10, 10, 7, 21)

# Prepare titles for the display
image_titles = [
    'Original', 'Normalized', 'Horizontal Flip', 'Rotated', 'Brightness Adjusted',
    'Contrast Enhanced', 'Zoomed', 'Saturation Adjusted', 'Hue Adjusted',
    'Grayscale', 'Edge Detection', 'Histogram Equalized', 'Gaussian Blurred', 'Noise Removed'
]

# Store all the processed images in a list
processed_images = [
    image_rgb, normalized_image, horizontal_flip, rotated_image, bright_image,
    contrast_image, zoomed_image, saturated_image, hue_adjusted_image, grayscale_image,
    edges_image, histogram_equalized_image, blurred_image, denoised_image
]

# Display all processed images in a grid
plt.figure(figsize=(15, 10))
for i in range(len(processed_images)):
    plt.subplot(3, 5, i + 1)
    plt.title(image_titles[i])
    plt.axis('off')
    # Display grayscale images with a gray colormap
    if len(processed_images[i].shape) == 2:
        plt.imshow(processed_images[i], cmap='gray')
    else:
        plt.imshow(processed_images[i])

plt.tight_layout()
plt.show()
