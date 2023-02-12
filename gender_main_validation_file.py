from keras.models import load_model
import cv2
import numpy as np

model = load_model('Trainingdataset/model-019.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

labels_dict={0:'Male',1:'Female'}
color_dict={0:(0,0,255),1:(0,255,0)}

while(True):

    ret,img=cap.read()
    if(img is not None):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_clsfr.detectMultiScale(gray,1.3,3)  

        for (x,y,w,h) in faces:

            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(32,32))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,32,32,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]

            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


        cv2.imshow('Result',img)
        k=cv2.waitKey(1)

        if k==ord("q"):
            break

cap.release()
cv2.destroyAllWindows()