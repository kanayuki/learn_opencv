import cv2 as cv
import matplotlib.pyplot as plt

# %%
p1 = r"Y:\VIDEO\BEAUTYLEG_P\[Beautyleg]2021-05-26 No.11 ChiChi[1V462M]\11ChiChi.mp4"

cap = cv.VideoCapture(p1)
cv.namedWindow('ChiChi.mp4', cv.WINDOW_NORMAL)
# cv.resizeWindow('ChiChi.mp4', 1080, 720)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv.imshow('ChiChi.mp4', frame)
        k = cv.waitKey(1)
        print(k)
        if k == ord('q'):
            break
    else:
        break

cap.release()

cv.destroyAllWindows()
