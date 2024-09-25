#Import cv2
import cv2
import numpy as np

#Reading the Imagery
image = cv2.imread("images/Mountain.png")

# Checking if the Image is loaded successfully
if image is None:
    print("Error loading ")
else:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyWindow()

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
lower_red = np.array([0, 255, 100])
upper_red = np.array([10,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)

result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyWindow()