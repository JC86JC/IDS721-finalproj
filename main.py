from flask import Flask
from model import *

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# preprocess
X_train = train.drop(['id','price'],axis = 1)
y_train = train[['price']]
X_test = test.drop(['id'],axis = 1)
x_train1, x_test1 = preprocess(X_train,X_test)
y = train['price'].to_numpy().ravel()

app = Flask(__name__)


@app.route("/")
def home():
    html = f"<h3>Housing Price Prediction Home</h3>"
    return html.format(format)





@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction 
    input looks like:
            {
    "CHAS":{
      "0":0
    },
    "RM":{
      "0":6.575
    },
    "TAX":{
      "0":296.0
    },
    "PTRATIO":{
       "0":15.3
    },
    "B":{
       "0":396.9
    },
    "LSTAT":{
       "0":4.98
    }
    result looks like:
    { "prediction": [ 20.35373177134412 ] }
    """


    json_payload = request.json
    LOG.info(f"JSON payload: {json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    LOG.info(f"inference payload DataFrame: {inference_payload}")
    scaled_payload = scale(inference_payload)
    prediction = list(clf.predict(scaled_payload))
    return jsonify({'prediction': prediction})


@app.route('/cd')
def cd():
    return 'Welcome to the page for testing continuous delivery! Test 2.'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)