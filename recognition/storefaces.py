import cv2
import numpy as np
dataset = cv2.CascadeClassifier('data.xml.txt')
capture = cv2.VideoCapture(0)
faceData = []
while True:
    ret, img = capture.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 2)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (50, 50))
            if len(faceData) < 50:
                faceData.append(face)
                print(len(faceData))
        cv2.imshow('result', img)
        if cv2.waitKey(1) == 27 or len(faceData) >= 50:
            break
    else:
        print("camera not working")
faceData = np.asarray(faceData)
np.save("sahil.npy", faceData)
cv2.destroyAllWindows()
capture.release()