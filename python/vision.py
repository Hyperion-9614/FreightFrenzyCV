import cv2
import numpy as np

cap = cv2.VideoCapture(0)


def detectCube(frame):
    # Convert BGR to HSV
    hsv = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower = np.array([27, 26, 189])
    upper = np.array([40, 255, 255])
    masked = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=masked)
    ret, thresh_img = cv2.threshold(
        cv2.cvtColor(res, cv2.COLOR_BGR2GRAY),
        100,
        255,
        cv2.THRESH_OTSU + cv2.THRESH_BINARY,
    )
    contours, hierarchy = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) <= 0:
        return frame
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.line(
        frame,
        (int(frame.shape[1] / 3), 0),
        (int((frame.shape[1] / 3)), frame.shape[0]),
        (255, 0, 0),
        2,
    )
    cv2.line(
        frame,
        (int((frame.shape[1] / 3) * 2), 0),
        (int((frame.shape[1] / 3) * 2), frame.shape[0]),
        (255, 0, 0),
        2,
    )
    centerX = int((x + (x + w)) / 2)
    if centerX < int((frame.shape[1] / 3)):
        cv2.putText(frame, "Left", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif centerX > int((frame.shape[1] / 3) * 2):
        cv2.putText(frame, "Right", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(
            frame, "Center", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return frame


while True:
    ret, frame = cap.read()
    cv2.imshow("frame", detectCube(frame))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
