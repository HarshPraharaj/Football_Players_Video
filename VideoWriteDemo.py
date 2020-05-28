import cv2


cap = cv2.VideoCapture('ip.mp4')

while True:
    ret,frame = cap.read()

    cv2.imshow('output',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()