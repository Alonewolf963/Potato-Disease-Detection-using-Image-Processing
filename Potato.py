import cv2

cascPath = "D:\SPIT sem4\mini project\Xml file\Sprouted_xml_file.xml"
potatoCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

# fontScale
fontScale = 0.5

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 1

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    detected = potatoCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=12,
        minSize=(10, 10)
    )

    # Draw a rectangle around the Potatoes
    for (x, y, w, h) in detected:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frames, 'Sprouted Potato', (x,y-10), font,
                    fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
