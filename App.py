import os
import numpy as np
import librosa as lsa
import tensorflow as tf
from attr.validators import max_len
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = tf.keras.models.load_model('Models/audio_model.keras')

max_length = 56293

def preprocessing(audio_path, max_length):
    audio, sr = lsa.load(audio_path, sr=None)  # ✅ pass path, not FileStorage
    mfcc = lsa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T

    # Pad or truncate
    if mfcc.shape[0] < max_length:
        pad_width = max_length - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
    else:
        mfcc = mfcc[:max_length, :]

    return mfcc.reshape(1, max_length, 13)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save file first
    audio_path = os.path.join("uploads", audio_file.filename)
    audio_file.save(audio_path)

    # Pass file path to preprocessing
    paddedSam = preprocessing(audio_path, max_length)

    prediction = model.predict(paddedSam)

    predictedClass = np.argmax(prediction , axis=1)[0]

    confidence = float(prediction[0][predictedClass])

    result = "Fake" if predictedClass == 1 else "Real"

    return jsonify({'Prediction': result, 'Confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
