import cv2
import numpy as np
import  time
import os
import HandTrackModule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "Header"
myList = os. listdir(folderPath)
print(myList)
overlayList =[]
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (244, 235, 178)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:

    #1. 이미지 가져오기
    success, img = cap.read()
    img = cv2.flip(img,1)

    #2. 손 랜드마크 찾기
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) !=0:

        print(lmList)

        # 검지와 중지의 끝
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]


        #3. 어떤 손가락이 위로 올라가 있는지 확인
        fingers = detector.fingersUp()
        #print(fingers)

        #4. 선택 모드인 경우 - 두 손가락이 다 먹어버렸습니다.
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")
            # Checking for the click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1-15), (x2, y2+15), drawColor, cv2.FILLED)

        #5. 그리기 모드 - 집게 손가락이 위로
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

            # 모든 손가락이 위로 올라가면 캔버스 지우기
            # if all (x >= 1 for x in fingers):
            # imgCanvas = np.zeros((720, 1280, 3), np.uint8)

            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, imgCanvas)

    # 헤더 이미지 설정
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)