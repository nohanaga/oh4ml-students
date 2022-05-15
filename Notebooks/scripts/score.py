import pickle
import os, json
import numpy
from azureml.core.model import Model
from statsmodels.tsa.arima_model import ARIMA

# Sample Test Script

# step_size=[3]
# test_sample = json.dumps({"data": step_size})
# test_sample = bytes(test_sample, encoding="utf8")
# prediction = service.run(input_data=test_sample)
# print(prediction)

def init():
    global model
    import joblib

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'models/arima_model.pkl')
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = numpy.array(data)
        result=model.forecast(steps=data[0])[0]
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
