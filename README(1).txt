Project Title:- Image CLassification using CNN
This Project Demonstrate Real-Time Image Classification using Convolutional Neural Network using Pre-Trained Model like MobileNetV2.

Goal:-Capture Live Video from your Webcam and classify each frame is one of the ImageSet Categories like(Dog, cat,keyboard etc.)
Core Idea:- Used Pre trained CNN model MobileNetV2 to recognized object without having to train from scratch.
OutPut:- Display the webcam feed with a label (class name + Confidence Percentage) overlaid in real-time.

Description:-
	In this Project we use your webcam along with Pre-Trained Convolutional Neural Network Model MobileNetV2 to classify objects in real-time.This System will capture video frames ,process them and display the predicted object label confidence percentage on the screen.

This will help You understand:
	*How CNN works for image recognization.
	*The concept of transfer learning using Pre-Trained models.
	*Integration of OpenCV(cv2) with DL Models.

Dependencies :- 
Install required Python package before running project.
PIP install tensorflow.
pip install opencv-python

Pre-Trained Model Name:- MobileNetV2

Source Code:-
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter all
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input,decode_predictions

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
def main():
    MarvellousImageClassifier()

if __name__ == "__main__":
    main()


Explaination of Project and work flow of Project:
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = filter all
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input,decode_predictions

. cv2 (OpenCV) :- For Webcam capture, image processing and display.
. numpy :- For Numerical operations.(Reshape, Expand dimensions)
. MobileNetV2 + utils :- Pre-Trained CNN model+preprocessing + decoding predictions

# Loading Data From Pre-trained model.
 model = MobileNetV2(weights="imagenet")
 .Load MobileNetV2 a lightweight CNN trained on ImageNet (1.4M images, 1000 classes)
 .Since its pre-trained , you dont need to train it again - its ready for inference.

# Accessing Webcam
  cap = cv2.videoCapture(0)
  . Opens the default webcam (index 0)
  . If webcam isnot found, it shows an error.

# Frame Capture and Preprocessing:-
  ret,frame = cap.read()
  img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img,(224,224))
  x = np.expand_dims(img_resized,axis=0).astype(np.float32)
  x = preprocess_input(x)
  
  . Frame:- Captures one video frame.
  . BGR -> RGB :- OpenCV uses BGR, but MobileNet expects RGB.
  . Resize:- CNN input size = 224*224.
  . Expand Dimensions :- CNN expects input shape (1,224,224,3).
  . Preprocess:- Normalized pixel for MobileNetV2.

# Prediction:-
  preds = model.predict(x, verbose = 0)
  decoded = decode_predictions(preds,top=1)[0][0]  #(class_id,class_name,score)
  label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

  .Model.predict:- Runs Forward Pass through CNN.
  .decode_prediction:- Convert raw Model Output into readable labels (ex. "Labrador    retriever").
  .Top-1 Label:- Takes the best match with confidence percentage.

# Overlay Prediction:-
  cv2.putText(frame,label,(16,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2,cv2.LINE_AA)
  cv2.imshow("Real-Time CNN Classification  (MobileNetV2)",frame)
  
 . Draws the prediction text on the video frame.
 . Displays the live webcam feed with classification results.

# Exit and cleanup:- 
          if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

 .Press "Q" to quit.
 . Release webcam and closes all windows.

# Important Concept

 . CNN(Convolutional Neural Network):- Learns spatial features from images (edges-> textures->objects).
 . Tranfer Learning:- Using a model trained on a huge dataset(ImageNet) for your own application.
 . MobileNetV2 :- A lightwight CNN optimized for speed and accuracy , good for real-time applications.

Author 
 Sohan Tejas Vatane
 Date : 29/09/2025
 Day :  Monday
