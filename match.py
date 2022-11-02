import cv2 as cv

sift = cv.xfeatures2d.SIFT_create()

img1 = cv.imread(r"C:\Users\HuiLing\Desktop\beauty.png")
img2 = cv.imread(r"C:\Users\HuiLing\Desktop\tile.png")

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

matcher = cv.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
ms = matcher.knnMatch(des2, des1, 2)
good = list(filter(lambda t: t[0].distance < 0.7 * t[1].distance, ms))
print(ms)
print('========================')
print(good)
ret = cv.drawMatchesKnn(img2, kp2, img1, kp1, good, None)

cv.imshow('match', ret)
cv.waitKey()
