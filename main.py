from flask import Flask
from model import *
from sklearn.metrics import confusion_matrix

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


@app.route("/")
def home():
    html = "<h3>Housing Price Prediction Home</h3>"
    return html.format(format)

@app.route("/report")
def report(clf, x_test, y_test):
    '''Print prediction metric on test.csv'''
    y_hat = clf.predict(x_test)
    
    cm = confusion_matrix(y_test, y_hat)
    acc = clf.score(x_test, y_test)
    print('confusion matrix:', '\n', cm, '\n', '='*80,  'prediction accuracy', acc)

@app.route('/html')
def html():
    """Returns some custom HTML"""
    return """
    <title>This is a Hello World World Page</title>
    <p>Hello</p>
    <p><b>World</b></p>
    """


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)