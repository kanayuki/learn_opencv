import cv2

faceCascade = cv2.CascadeClassifier(
    r"C:\Users\HuiLing\anaconda3\envs\opencv\Library\etc\haarcascades\haarcascade_frontalface_alt2.xml")
# path = r"E:\Image\Twitter\Fb4-LKDacAUbgvr.jfif"
path = r"E:\Image\Twitter\FW8tvQlUcAAS-lO.jfif"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale2(gray, 1.01, 5,0,(100,100),(500,500))
if faces is None:
    print('没有找到人脸！')

print(faces)
for (x, y, w, h) in faces[0]:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.namedWindow('Beauty', cv2.WINDOW_NORMAL)
cv2.imshow('Beauty', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
