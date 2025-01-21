from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load a pre-trained fake profile detection model (replace with your model)
model = tf.keras.models.load_model('fake_profile_model.h5')

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image.resize((224, 224))) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    result = "Fake Profile Detected" if prediction[0][0] > 0.5 else "Profile Seems Legitimate"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)