import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import pickle

#############################################

threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# IMPORT THE TRANNIED MODEL
#pickle_in=open("last_model_trained_.h5","rb")  ## rb = READ BYTE
#model=pickle.load(pickle_in)
model = load_model('last_model_trained_.h5')

def preprocessing(img):
    img = img/255
    return img

code = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

def getClassName(classNo) :
    for i,j in code.items() :
        if j == classNo :
            return i



    # PROCESS IMAGE

imgOrignal = cv2.imread('car2.jpeg')  # reads image 'opencv-logo.png' as grayscale
plt.imshow(imgOrignal)
img =cv2.imshow('hadik',imgOrignal)
img = np.asarray(imgOrignal)
img = cv2.resize(img, (32,32),interpolation=cv2.INTER_CUBIC)
img = preprocessing(img)

cv2.imshow("Processed Image", img)
img = img.reshape(1, 32, 32, 3)
cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
predictions = model.predict(img)
classIndex = model.predict_classes(img)
probabilityValue =np.amax(predictions)
print(getClassName(classIndex))
cv2.putText(imgOrignal,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow("Result", imgOrignal)
cv2.waitKey(0)