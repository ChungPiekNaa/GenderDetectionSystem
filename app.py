from flask import *
from flask_cors import CORS
import onnxruntime
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from waitress import serve
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tif', 'tiff'}

# Load the classes
with open('classes.txt', 'r') as file:
    classes = [line.strip() for line in file.readlines()]

# Load the ONNX model
onnx_model_path = 'model.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        # Read the input image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a PIL image from the NumPy array
        image = Image.fromarray(image)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).numpy()  # Convert to NumPy array

        # Perform inference
        output = onnx_session.run(None, {'input.1': input_batch})  # Update the input name

        # Apply softmax to get probabilities
        probabilities = np.exp(output[0][0]) / np.sum(np.exp(output[0][0]))

        # Get the predicted class and confidence score
        predicted_class = int(np.argmax(probabilities))
        confidence = probabilities[predicted_class]

        result = {
            "predicted_class": classes[predicted_class],
            "confidence": float(confidence)
        }

        return jsonify(result)

    else:
        return jsonify({"error": "Invalid file extension"})

if __name__ == '__main__':
    # Set up ngrok tunnel using bcs22020008@student.uts.edu.my account.
    ngrok.set_auth_token("2wkeDUDlGA9M1BnxdW2gK7Jyo1K_35NoxCxA4d9U72F73eG9U") # Get this authentication from your NGROK account.
    public_url = ngrok.connect(8006, domain="bluebird-first-multiply.ngrok-free.app") # Get this domain from your NGROK account. Use this to connect to the website.
    print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:8006")

    # Start Flask server with waitress
    serve(app, host='0.0.0.0', port=8006) # if you use app.run, you must set debug=False. Else will get error.

