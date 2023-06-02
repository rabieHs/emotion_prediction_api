import cv2
from keras.models import model_from_json
from flask import Flask, request, jsonify
import numpy as np
import tempfile

app = Flask(__name__)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

@app.route('/predict', methods=['POST'])
def predict():
    video_bytes = request.data
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(video_bytes)
        f.flush()
        cap = cv2.VideoCapture(f.name)
    while True:
        maxindex=0
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))

    # return the emotion prediction
    return jsonify(emotion=emotion_dict[maxindex])

if __name__ == '__main__':
    app.run()