from flask import Flask,render_template,request
from logging import FileHandler,WARNING

import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('disaster_model3.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # making prediction
    return render_template('result.html', prediction_text='Entered features classify the Disaster type as {}.' .format(prediction[0]))
    

@app.errorhandler(500)
def internal_error(error):
    return "500 error"

    
if __name__=='__main__':
    app.run(port=8000)