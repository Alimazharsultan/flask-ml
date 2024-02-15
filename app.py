from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'plant_leaf_diseases_model.h5'
model = load_model(MODEL_PATH)

@app.route('/')
def upload_form():
    return '''
    <!DOCTYPE html>
    <html>
    <body>

    <h2>Upload Image for Prediction</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
      Select image to upload:
      <input type="file" name="file" id="file">
      <input type="submit" value="Upload Image" name="submit">
    </form>

    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read the image file to PIL Image
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image and prepare it for classification
        processed_image = prepare_image(image, target_size=(256, 256))  # Adjust target_size as per your model's requirement

        # Predict
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        print('PredictedClass', predicted_class)

        # Return the result
        return jsonify({'predicted_class': str(predicted_class[0])})

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == '__main__':
    app.run(debug=True)