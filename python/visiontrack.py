import cv2
import numpy as np

def nothing(x):
    pass

# Make track bar windows
cv2.namedWindow('result1')
h1, s1, v1 = 100, 100, 100
img_low = np.zeros((20,300,3),np.uint8)

cv2.namedWindow('result2')
h2, s2, v2 = 100, 100, 100
img_high = np.zeros((20,500,3),np.uint8)

# Make track bar
cv2.createTrackbar('h1','result1', 0, 255, nothing)
cv2.createTrackbar('s1','result1', 0, 255, nothing)
cv2.createTrackbar('v1','result1', 0, 255, nothing)

cv2.createTrackbar('h2','result2', 0, 255, nothing)
cv2.createTrackbar('s2','result2', 0, 255, nothing)
cv2.createTrackbar('v2','result2', 0, 255, nothing)

# cap = cv2.VideoCapture("http://192.168.1.242:4747/video")
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
def detectCube(frame):
    frame = frame[326:960, 0:1280]
    # Convert BGR to HSV
    hsv = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # print('dimensions: ', frame.shape)
    
    # trackbars to edit range
    cv2.imshow('result1', img_low)
    h1 =  cv2.getTrackbarPos('h1','result1')
    s1 =  cv2.getTrackbarPos('s1','result1')
    v1 =  cv2.getTrackbarPos('v1','result1')
    img_low[:] = [h1,s1,v1]
    
    cv2.imshow('result2', img_high)
    h2 =  cv2.getTrackbarPos('h2','result2')
    s2 =  cv2.getTrackbarPos('s2','result2')
    v2 =  cv2.getTrackbarPos('v2','result2')
    img_high[:] = [h2,s2,v2]

    # define range of blue color in HSV
    # lower = np.array([27, 26, 189])
    # upper = np.array([40, 255, 255])
    lower = np.array([h1,s1,v1])
    upper = np.array([h2,s2,v2])
    print('lower: ' , lower)
    print('upper: ' , upper, '\n')
    # home test: lower: [ 18  71 132] and upper: [77 255 255]

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

    # image_masked = cv2.bitwise_and(frame, frame, mask = masked)
    return [frame, res, masked]


while True:
    ret, frame = cap.read()
    cv2.imshow("frame", detectCube(frame)[0])
    cv2.imshow("bitwise and", detectCube(frame)[1])
    cv2.imshow("masked", detectCube(frame)[2])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
