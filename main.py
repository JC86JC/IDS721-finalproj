from flask import Flask
from model import *

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# preprocess
X_train = train.drop(['id','price'],axis = 1)
y_train = train[['price']]
X_test = test.drop(['id', 'price'],axis = 1)
x_train1, x_test1 = preprocess(X_train,X_test)
y_train = train['price'].to_numpy().ravel()
y_test = test['price'].to_numpy().ravel()

#fit model
clf = model(x_train1,y_train)

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World! This is my first flask app.'

@app.route('/name/<value>')
def name(value):
    """parameter"""
    return "The name you entered: %s" % value

@app.route('/cd')
def cd():
    return 'Welcome to the page for testing continuous delivery! Test 2.'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)