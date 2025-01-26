# Imports
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

car_image = imread("test_images/car6.jpg", as_gray=True) # Returns 2D array representing grey scale pixels

# skimage returns ranges 0:1 multiply by 255 for common 0:255 grey scale range.
grey_car_image = car_image * 255

### Plotting
fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1 is the true grey scale photo
ax1.imshow(grey_car_image, cmap="gray")
# ax2 is a binary (black or white) photo
threshold_value = threshold_otsu(grey_car_image)
binary_car_image = grey_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()

