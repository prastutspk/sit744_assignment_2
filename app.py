from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model
model = tf.keras.models.load_model('best_model.h5')

# Initialize Flask app
app = Flask(__name__, static_url_path='/static')

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image file from the request
        image = request.files['image']
        
        # Read the image file as a PIL image
        img = Image.open(image)
        
        # Resize the image to match the input requirements of the model
        img = img.resize((64, 64))
        
        # Convert the image to a NumPy array
        img_array = np.array(img)
        # Preprocess the image
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Expand the dimensions to match the input shape of the model
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction using the loaded model
        prediction = model.predict(img_array)
        
        # Get the predicted class label
        class_label = np.argmax(prediction)

        print(class_label)
        
        return render_template('result.html', class_label=class_label)
    
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run()
