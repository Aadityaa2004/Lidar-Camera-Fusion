import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Couldn't open Camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not read")
        break

    cv.imshow('Camera Feed', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()