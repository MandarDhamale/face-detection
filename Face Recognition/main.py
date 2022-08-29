import cv2
from simple_facerec import SimpleFacerec



sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


#loading camera
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while (True):

    ret, frame = cap.read()

    #detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,200,0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,0), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()