import cv2

img = cv2.imread("/Users/aditya/Programming/FTC/yellow.jpg")
cv2.line(img, (int(img.shape[0]/2, 0)), (int(img.shape[0]/2), img.shape[1]), (0, 0, 255), 5)
print(img.shape[0])
while True:
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break