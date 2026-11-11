import cv2
import numpy as np

circles = np.zeros((4, 2), np.int32)
counter = 0


def mousePoint(event, x, y, flag, param):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter < 4:
            circles[counter] = x, y
            counter += 1
            print(circles)


def order_points(pts):
    """Return points ordered as: top-left, top-right, bottom-left, bottom-right"""
    pts = np.float32(pts)
    ordered = np.zeros((4, 2), np.float32)

    s = pts.sum(axis=1)             # x + y
    d = np.diff(pts, axis=1)        # y - x

    ordered[0] = pts[np.argmin(s)]  # top-left     (smallest x + y)
    ordered[3] = pts[np.argmax(s)]  # bottom-right (largest x + y)
    ordered[1] = pts[np.argmin(d)]  # top-right    (smallest y - x)
    ordered[2] = pts[np.argmax(d)]  # bottom-left  (largest y - x)
    return ordered


img = cv2.imread('resources/photos/3.png')
if img is None:
    raise FileNotFoundError("Image not found - check the path")
imgOriginal = img.copy()

cv2.namedWindow('original image')
cv2.setMouseCallback('original image', mousePoint)  # set once, outside the loop

while True:
    img = imgOriginal.copy()

    if counter == 4:
        width, height = 250, 350
        pts1 = order_points(circles)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(imgOriginal, matrix, (width, height))
        cv2.imshow('Output image', imgOutput)

    for x in range(counter):
        cv2.circle(img, (int(circles[x][0]), int(circles[x][1])), 3, (0, 255, 0), cv2.FILLED)

    cv2.imshow('original image', img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):           # reset to pick new points
        counter = 0
        circles[:] = 0
        cv2.destroyWindow('Output image')

cv2.destroyAllWindows()