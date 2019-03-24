import cv2
import numpy as np
import math

# img = cv2.imread('hand1.png')
img = cv2.imread('hand2.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Define range of skin
lower = np.array([0, 20, 70], dtype=np.uint8)
upper = np.array([20, 255, 255], dtype=np.uint8)

# Extract skin color to a mask
mask = cv2.inRange(hsv, lower, upper)
# Reduce noise
mask = cv2.medianBlur(mask, 5)
# Find contours
_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Chose max contour area
max_contour = max(contours, key=lambda x: cv2.contourArea(x))

# Approx the contour
for c in contours:
    count = 1
    max_contour = c
    approx = cv2.approxPolyDP(max_contour, 0.0004 * cv2.arcLength(max_contour, True), True)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    M = cv2.moments(c)
    if M['m00'] >= 2000:
        if M['m00'] != 0:
            cx = int((M["m10"] / M["m00"]))
            cy = int((M["m01"] / M["m00"]))
            # Calc fingers
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    # Calc distance 3 points in triangle
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    p = (a + b + c) / 2
                    s = math.sqrt(p * (p - a) * (p - b) * (p - c))
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57  # to degree
                    if angle <= 90:
                        count += 1
                        cv2.circle(img, far, 3, [255, 0, 0], -1)
                    cv2.line(img, start, end, [0, 255, 0], 3)

                cv2.putText(img, str(count), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 50), 3,
                            cv2.LINE_AA)

cv2.imshow('test', mask)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()