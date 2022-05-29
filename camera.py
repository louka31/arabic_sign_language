import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model



class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()

        model = load_model('augmentation.h5')
        # Load class names
        f = open('gesture.names', 'r')
        classNames = f.read().split('\n')
        f.close()
      
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)

        roi = frame[y1:y2, x1:x2]
        roi =cv2.resize(roi,(64,64))
        roi= cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

        roi=roi/255.0
    
        roi=cv2.flip(roi, 1)

        landmarks = []
    
        landmarks.append(roi)
        prediction = model.predict(np.array(landmarks))
        
        print(prediction)
        
        classID = np.argmax(prediction)
        print(classID)
        if(classID >= 0 and classID <= 31):
            className = classNames[classID]
        
        if(classID >= 0 and classID <= 31):
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)     
        else:
            cv2.putText(frame, 'None', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)     
            

        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()