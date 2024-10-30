import cv2

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    cv2.imshow("video", frame)
    print(f"{frame.shape[1]}, {frame.shape[0]}")

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()