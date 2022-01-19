#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dependencies
from flask import Flask, request, jsonify
#from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if modelo:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)
            print(query)
            prob_predictions = modelo.predict_proba(query)
            yhat = prob_predictions[:, 1]
            y_pred_best = (prob_predictions[:,1] >= 0.018339461919462164).astype(int)
               
            return jsonify({'prediction_best': str(y_pred_best)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = '12345' 

    modelo = joblib.load("modelo.sav") # Load "model.pkl"
    model_columns = joblib.load('model_columns.sav')
    print ('Model loaded')
    app.run(port=port)

