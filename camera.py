import cv2 as cv

faceCascade = cv.CascadeClassifier(
    r"C:\Users\HuiLing\anaconda3\envs\opencv\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml")

# cap = cv.VideoCapture(r"Y:\VIDEO\BEAUTYLEG_P\[Beautyleg]2021-05-26 No.11 ChiChi[1V462M]\11ChiChi.mp4")
cap = cv.VideoCapture(0)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
fourcc = int(cap.get(cv.CAP_PROP_FOURCC))

print(f'{width}x{height}, FPS:{fps}, FOURCC:{fourcc}')

cv.namedWindow('USB Camera', cv.WINDOW_NORMAL)
cv.namedWindow('mask', cv.WINDOW_NORMAL)

mog = cv.bgsegm.createBackgroundSubtractorMOG()
threshold1 = 50
threshold2 = 150


def update_edge(value):
    global threshold1,threshold2
    threshold1 = cv.getTrackbarPos('min', 'mask')
    threshold2 = cv.getTrackbarPos('max', 'mask')


cv.createTrackbar('min', 'mask', 50, 200, update_edge)
cv.createTrackbar('max', 'mask', 50, 200, update_edge)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # mask = mog.apply(frame)
        mask = cv.Canny(frame, threshold1, threshold2)
        cv.imshow('mask', mask)

        faces = faceCascade.detectMultiScale(gray, 1.1, 4, 0, (100, 100), (300, 300))
        if faces is None:
            print('没有找到人脸！')

        print(faces)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.putText(frame, "YUKI", (x, y), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        cv.imshow('USB Camera', frame)

        key = cv.waitKey(5)
        # print(key)
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
