from base64 import decode
from cProfile import label
from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load_model('cnn_pneumonia.hdf5')

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
    return render_template('index.html')


@app.route('/submit', methods=['GET','POST'])
def predict():
    if request.method =='POST':
        imagefile = request.files['imagefile']
        image_path = 'static/' + imagefile.filename
        imagefile.save(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224,224)) #change input size
        img = np.reshape(img, (1, 224,224,3))
        img = np.array(img)

        #predict percentage
        max=0
        pred = model.predict(img)[0]
        for i in pred:
            if max is None or i>max:
                max = i
        pred = round(max, 4)
        
        # label showing
        pred_label = model.predict(img)
        pred_label = np.argmax(pred_label)
        dataset = pd.read_csv('labels.csv')
        labels = dataset.Labels.tolist()

        pred_label = labels[pred_label]
    
    return render_template('index.html', prediction=pred, prediction_label=pred_label, image_path=imagefile.filename)

    # return render_template('index.html', prediction = 0, image_path=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)