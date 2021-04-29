from flask import Flask, request, render_template, session, redirect
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle

# load data
test = pd.read_csv("test.csv")

# preprocess
x_test1 = pd.read_csv('processed_test.csv')
y_test = test['price'].to_numpy().ravel()


#load model
clf = pickle.load( open('model.pkl','rb'))

app = Flask(__name__)


@app.route("/")
def home():
    html = "<h3>Housing Price Prediction Home</h3>"
    return html.format(format)

@app.route("/report")
def report():
    '''Print prediction metric on test.csv'''
    y_hat = clf.predict(x_test1)
    
    cm = confusion_matrix(y_test, y_hat)
    cm = pd.DataFrame(cm, columns=['TrueClass_1','TrueClass_2','TrueClass_3','TrueClass_4'], index=['PredictClass_1','PredictClass_2','PredictClass_3','PredictClass_4'])
    #acc = clf.score(x_test, y_test)
    #print('confusion matrix:', '\n', cm, '\n', '='*80,  'prediction accuracy', acc)
    return cm.to_html(header="true", table_id="Confusion_matrix")

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
