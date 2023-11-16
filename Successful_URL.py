import cv2
import urllib.request
import numpy as np

cascPath = "D:\SPIT sem4\mini project\Xml file\Yogendra_xml_2.xml"
PotatoCascade = cv2.CascadeClassifier(cascPath)

url = 'http://192.168.1.101/capture?'
#'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
cv2.namedWindow("Potato Testing", cv2.WINDOW_AUTOSIZE)

font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.6

# Red color in BGR
color = (0, 0, 255)

# Line thickness of 1 px
thickness = 1

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Potato = PotatoCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    for x, y, w, h in Potato:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(img, 'Sprouted', (x, y - 10), font,
                    fontScale, color, thickness, cv2.LINE_AA)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

    cv2.imshow("Detected Potatoes", img)
    key = cv2.waitKey(4)
    if key == ord('q'):
        break

cv2.destroyAllWindows()