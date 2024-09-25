#Import cv2

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading the Image
image = cv2.imread("images/Mountain.png")

# Checking if the Image is loaded successfully
if image is None:
    print("Error loading image")
else:
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Mountain Image")
    plt.xlabel("Horizontal Grid")
    plt.ylabel("Vertical Grid")
    plt.show()
