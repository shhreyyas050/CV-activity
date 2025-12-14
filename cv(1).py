import cv2
import numpy as np
import os

# -----------------------------
# Load image (CHANGE PATH)
# -----------------------------
img_path = "shreyas.jpg"
img = cv2.imread(img_path)

if img is None:
    print(f"Image not found! Update path or place '{img_path}' next to the script.")
    raise SystemExit(1)

cv2.imshow("Original Image", img)
cv2.waitKey(0)

# -----------------------------
# 1. Convert to Grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)

# -----------------------------
# 2. Save image in different format
# -----------------------------
cv2.imwrite("saved_image.png", img)

# -----------------------------
# 3. Resize image
# -----------------------------
resized = cv2.resize(img, (400, 300))
cv2.imshow("Resized Image", resized)
cv2.waitKey(0)

# -----------------------------
# 4. Flip image horizontally & vertically
# -----------------------------
h_flip = cv2.flip(img, 1)
v_flip = cv2.flip(img, 0)
cv2.imshow("Horizontal Flip", h_flip)
cv2.imshow("Vertical Flip", v_flip)
cv2.waitKey(0)

# -----------------------------
# 5. Crop ROI (safely clipped to image bounds)
# -----------------------------
h, w = img.shape[:2]
y1, y2 = 100, 300
x1, x2 = 150, 350
y1 = max(0, min(y1, h))
y2 = max(0, min(y2, h))
x1 = max(0, min(x1, w))
x2 = max(0, min(x2, w))
roi = img[y1:y2, x1:x2]
if roi.size == 0:
    print("ROI is empty; skipping display.")
else:
    cv2.imshow("Cropped ROI", roi)
    cv2.waitKey(0)

# -----------------------------
# 6. Draw shapes
# -----------------------------
shapes = img.copy()
cv2.rectangle(shapes, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.circle(shapes, (300, 150), 50, (255, 0, 0), 2)
cv2.line(shapes, (0, 0), (400, 300), (0, 0, 255), 2)
cv2.imshow("Shapes", shapes)
cv2.waitKey(0)

# -----------------------------
# 7. Put custom text
# -----------------------------
text_img = img.copy()
cv2.putText(text_img, "Shrey", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.imshow("Text Image", text_img)
cv2.waitKey(0)

# -----------------------------
# 8. Convert to HSV & extract red color (covers both red ranges)
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
red_result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Red Color Extraction", red_result)
cv2.waitKey(0)

# -----------------------------
# 9. Split BGR channels
# -----------------------------
b, g, r = cv2.split(img)
cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)
cv2.waitKey(0)

# -----------------------------
# 10. Merge channels
# -----------------------------
merged = cv2.merge([b, g, r])
cv2.imshow("Merged Image", merged)
cv2.waitKey(0)

# -----------------------------
# 11. Increase brightness
# -----------------------------
bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
cv2.imshow("Bright Image", bright)
cv2.waitKey(0)

# -----------------------------
# 12. Adjust contrast
# -----------------------------
contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
cv2.imshow("High Contrast", contrast)
cv2.waitKey(0)

# -----------------------------
# 13. Gaussian Blur
# -----------------------------
gaussian = cv2.GaussianBlur(img, (7, 7), 0)
cv2.imshow("Gaussian Blur", gaussian)
cv2.waitKey(0)

# -----------------------------
# 14. Median Blur
# -----------------------------
median = cv2.medianBlur(img, 7)
cv2.imshow("Median Blur", median)
cv2.waitKey(0)

# -----------------------------
# 15. Rotate 90, 180, 270 degrees
# -----------------------------
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img, cv2.ROTATE_180)
rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("Rotate 90", rot90)
cv2.imshow("Rotate 180", rot180)
cv2.imshow("Rotate 270", rot270)
cv2.waitKey(0)

cv2.destroyAllWindows()

