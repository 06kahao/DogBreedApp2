from datetime import datetime
from flask import render_template, request, jsonify, Flask, current_app
from scipy.misc import imresize
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from glob import glob
from PIL import Image
from os import environ
import numpy as np
import io
import sys
import base64
from load import *

app = Flask(__name__)
#with app.app_context():
    #print(current_app.name)

global model, graph
model, graph = init()

# load list of dog names 
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='BreedID App',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Something special for you.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Xception Transfer Learning.'
    )

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded))
    x = img.resize((299,299))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with graph.as_default():
        pred_vector = model.predict(x)
        breed = dog_names[np.argmax(pred_vector)]
        response = {'prediction': breed}
        return jsonify(response)

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
