
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2

from matplotlib import pyplot as plt


#location of image
image = cv2.imread('/home/rkaahean/examreco/test.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 1)
edged = cv2.Canny(~th, 175, 200)




cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.09 * peri, True)
 
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break


output = four_point_transform(image, displayCnt.reshape(4, 2))
warped = four_point_transform(th, displayCnt.reshape(4, 2))

#cv2.drawContours(image, [displayCnt], -1, (0, 255, 0), 2)
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

#cv2.imshow("output", edged)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

imgs = []
# loop over the digit area candidates
for cnt in cnts:
	# compute the bounding box of the contour
	rect = cv2.minAreaRect(cnt)
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)



cv2.drawContours(image, displayCnt, -1, (0, 255, 0), 2)
#cv2.imshow("output", imgs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
