import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance.Healthinsurance import HealthInsurance
import os

# carregando modelo
model = pickle.load(open('src/models/xgboost_model.pkl', 'rb'))

# inicializando API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    #testando dados recebidos
    if test_json: # há dados
        if isinstance(test_json, dict): # única linha
            test_raw = pd.DataFrame(test_json, index=[0])
        else: #múltiplas linhas
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # instanciando classe HealthInsurance
        pipeline = HealthInsurance()

        #preparando dados
        df1 = pipeline.data_preparation(test_raw)

        #previsão
        df_response = pipeline.get_prediction(model, test_raw, df1)

        return df_response

    else:
        return Response('{}', status=200, mimetype= 'application/json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host = '0.0.0.0', port = port, debug=True)
            