from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
import numpy as np
import time

app = Flask(__name__)
model = load_model('model_predictv4.h5')

# Fungsi untuk prediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    # Praproses gambar
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    start_time = time.time()
    predictions = model.predict(img_array)
    end_time = time.time()

    pred_label = np.argmax(predictions)
    pred_prob = predictions[0][pred_label]

    pred_time = round(end_time - start_time, 3)
    return pred_label, pred_prob, pred_time

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        file_path = 'static/Upload/' + file.filename
        file.save(file_path)
        pred_label, pred_prob, pred_time = predict_image(file_path)
        label_mapping = {0: 'paper', 1: 'rock', 2: 'scissors'}  # Ganti sesuai dengan kelas model Anda
        pred_label_str = label_mapping.get(pred_label, 'Unknown Class')
        return render_template('result.html', file_path=file_path, pred_label=pred_label_str, pred_prob=pred_prob, pred_time=pred_time)

if __name__ == '__main__':
    app.run(debug=True)