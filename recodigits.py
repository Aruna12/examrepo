import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import pickle

def convert_image(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	blkWhite = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 1)
	edged = cv2.Canny(~blkWhite, 175, 200)
	return edged, blkWhite

def getMarks(edged):
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
		# the marks
		if len(approx) == 4:
			displayCnt = approx
			break
	return displayCnt
def cleanup(blkMarks):
	thresh = cv2.threshold(blkMarks, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	return thresh

def show_image(text, image):
	cv2.imshow(text, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getNumbers(image, output):
	cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
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
		imgs += [output[y:y+h,x:x+w]]
	return imgs


image = cv2.imread('/home/rkaahean/examreco/test.jpg')

edged, blkWhite = convert_image(image)
displayCnt = getMarks(edged)

 
marks = four_point_transform(image, displayCnt.reshape(4, 2))
blkMarks = four_point_transform(blkWhite, displayCnt.reshape(4, 2))

cv2.drawContours(image, [displayCnt], -1, (0, 255, 0), 2)

thresh = cleanup(blkMarks)

imgs = getNumbers(thresh, marks)
_, finalImg = convert_image(imgs[2]) #Need to pass negation of finalImg 

finalImg = cleanup(finalImg)
img = cv2.resize(finalImg, (28, 28))
cv2.imwrite("num.png", img)

#show_image("text", img)
x, y = img.shape

k = 0
modimg = []
for i in range(0, 28):
	for j in range(0, 28):
		modimg.append(img[i, j])




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import joblib

# Importing the dataset
#dataset = pd.read_csv('train.csv')
#X = dataset.iloc[:, 1:].values
#y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
fn = '/home/rkaahean/examreco/pre.pkl'
clf = joblib.load(fn)
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, y_train)
#_ = joblib.dump(classifier, fn, compress = 9)

modimg = np.array(modimg)
modimg = modimg.reshape(1, -1)
y_pred = clf.predict(modimg)
print(y_pred)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

