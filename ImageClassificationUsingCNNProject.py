#########################################################################################
# Required Python packages
#########################################################################################
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter all
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input,decode_predictions

#########################################################################################
# Function Name : MarvellousImageClassifier
# Description : Take Images from pre Trained CNN Model
# Input : Open WebCam and capture Video
# output : Gives the Image Prediction in front of cameras
# Author : Tejas Khandu Vatane
# Data : 29/09/2025
#########################################################################################
def MarvellousImageClassifier():
    #1) Load Pre trained cnn (ImageNet)
    model = MobileNetV2(weights="imagenet")
    #2) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: COuld Not open Webcam..")
        return
    
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Error: Could Not read Frame...")
            break

        #3) Preprocess for MobileNetV2 : BGR -> RGB, resize to 224*224 , scale
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img,(224,224))
        x = np.expand_dims(img_resized,axis=0).astype(np.float32)
        x = preprocess_input(x)

        #4) Predict

        preds = model.predict(x, verbose = 0)
        decoded = decode_predictions(preds,top=1)[0][0]  #(class_id,class_name,score)
        label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

        #5) Overlay prediction on the frame
        cv2.putText(frame,label,(16,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Real-Time CNN Classification  (MobileNetV2)",frame)

        #6) EXIT
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    #7) CleanUp
    cap.release()
    cv2.destroyAllWindows()


#########################################################################################
# Function Name : main
# Description : Main Functions Where the execution starts
# Author : Tejas Khandu Vatane
# Data : 29/09/2025
#########################################################################################
def main():
    MarvellousImageClassifier()

#########################################################################################
# Application Starter
#########################################################################################
if __name__ == "__main__":
    main()