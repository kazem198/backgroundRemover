import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
# fpsReader = cvzone.FPS(avgCount=30)
imgBg = cv2.imread("images/images.jpg")
imgBg = cv2.resize(imgBg, (640, 480))
imgList = os.listdir("images")
path = "images"
images = []
for image in imgList:
    img = cv2.imread(f'{path}/{image}')
    img = cv2.resize(img, (640, 480))
    images.append(img)
# print(imgList)

imgNo = 0
while True:
    success, img = cap.read()

    imgOut = segmentor.removeBG(img, imgBg=images[imgNo], cutThreshold=0.1)

    # imgStacked = cvzone.stackImages([img, imgOut], cols=2, scale=1)
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)

    # _, img = fpsReader.update(imgStacked)

    cv2.imshow("imgStacked", imgStacked)
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    if key == ord("n"):
        if len(images) - 1 > imgNo:
            imgNo += 1
    if key == ord("p"):
        if imgNo > 0:
            imgNo -= 1
cap.release()
