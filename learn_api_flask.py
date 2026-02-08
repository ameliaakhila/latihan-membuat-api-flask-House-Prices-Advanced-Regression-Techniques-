from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('gbr_model.joblib')
feature_columns = joblib.load('feature_columns.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    datas = request.get_json()

    df = pd.DataFrame(datas['data'], columns=feature_columns[:len(datas['data'][0])])

    # Tambah kolom yang hilang
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(df)

    return jsonify({
        'prediction': prediction.tolist()
    })
if __name__ == '__main__':
    app.run(debug=True)