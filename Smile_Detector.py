import cv2

Classifier_Face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Classifier_Smile = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = Classifier_Face.detectMultiScale(frame_grayscale)
    

    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        the_face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smile = Classifier_Smile.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20)
        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40) , fontScale = 3,
            fontFace= cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))
    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()