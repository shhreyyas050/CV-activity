import cv2
import numpy as np

# -----------------------------
# Load Images (CHANGE PATHS)
# -----------------------------
img = cv2.imread("input.jpg")
logo = cv2.imread("logo.png")

if img is None:
    print("Main image not found!")
    exit()

cv2.imshow("Original Image", img)
cv2.waitKey(0)

# -----------------------------
# 6. Draw rectangle, circle & line
# -----------------------------
draw = img.copy()
cv2.rectangle(draw, (50, 50), (200, 200), (0, 255, 0), 2)
cv2.circle(draw, (300, 200), 50, (255, 0, 0), 2)
cv2.line(draw, (0, 0), (400, 300), (0, 0, 255), 2)
cv2.imshow("Shapes", draw)
cv2.waitKey(0)

# -----------------------------
# 7. Put custom text
# -----------------------------
text_img = img.copy()
cv2.putText(text_img, "Shrey", (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
cv2.imshow("Text on Image", text_img)
cv2.waitKey(0)

# -----------------------------
# 8. HSV & extract red color
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red1, upper_red1)
red = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("Red Color", red)
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
bright = cv2.convertScaleAbs(img, beta=60)
cv2.imshow("Brightened Image", bright)
cv2.waitKey(0)

# -----------------------------
# 12. Increase contrast
# -----------------------------
contrast = cv2.convertScaleAbs(img, alpha=1.8)
cv2.imshow("High Contrast", contrast)
cv2.waitKey(0)

# -----------------------------
# 13. Gaussian Blur
# -----------------------------
gaussian = cv2.GaussianBlur(img, (9, 9), 0)
cv2.imshow("Gaussian Blur", gaussian)
cv2.waitKey(0)

# -----------------------------
# 14. Median Blur
# -----------------------------
median = cv2.medianBlur(img, 9)
cv2.imshow("Median Blur", median)
cv2.waitKey(0)

# -----------------------------
# 15. Rotate 90, 180, 270
# -----------------------------
r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
r180 = cv2.rotate(img, cv2.ROTATE_180)
r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("Rotate 90", r90)
cv2.imshow("Rotate 180", r180)
cv2.imshow("Rotate 270", r270)
cv2.waitKey(0)

# -----------------------------
# 16. Rotate 45 degrees (no crop)
# -----------------------------
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rot45 = cv2.warpAffine(img, matrix, (w, h))
cv2.imshow("Rotate 45", rot45)
cv2.waitKey(0)

# -----------------------------
# 17. Translate image
# -----------------------------
M = np.float32([[1, 0, 50], [0, 1, 50]])
shifted = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Translated Image", shifted)
cv2.waitKey(0)

# -----------------------------
# 18. Binary Thresholding
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", binary)
cv2.waitKey(0)

# -----------------------------
# 19. Canny Edge Detection
# -----------------------------
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)

# -----------------------------
# 20. Overlay logo on image
# -----------------------------
if logo is not None:
    logo = cv2.resize(logo, (100, 100))
    img_overlay = img.copy()
    img_overlay[10:110, 10:110] = logo
    cv2.imshow("Overlay Image", img_overlay)
    cv2.waitKey(0)

cv2.destroyAllWindows()
