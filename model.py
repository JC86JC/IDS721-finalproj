import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.preprocessing import OrdinalEncoder, Normalizer, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle


def date_to_duration(df):
    timediff = pd.Timestamp.now() - pd.to_datetime(df)
    return [t.total_seconds()/(60*60*24) for t in timediff]

def date_to_cat(df):
    n = df.shape[0]
    month = np.empty(n,dtype = '<U32') 
    day = np.empty(n,dtype = '<U32') 
    year = np.empty(n,dtype = '<U32')
    for i in range(n):
        month[i],day[i],year[i] = X_train.loc[i,'last_review'].split("/") 
    return month,day,year

def preprocess(X, X_test, scale = True, year_as_num = True):
    x = X.copy()
    x_test = X_test.copy()
    
    date_variable = ['last_review','host_since']
    cat_variable = ['neighbourhood','room_type', 'host_is_superhost','bed_type','instant_bookable','is_business_travel_ready','cancellation_policy','require_guest_profile_picture','require_guest_phone_verification']
    
    if year_as_num == True:
        # transfer date to time difference
        for i in date_variable:
            x[i] = date_to_duration(x[i])
            x_test[i] = date_to_duration(x_test[i])
    else:
        # create new categorical variable
        for i in date_variable:
            x[i+"month"],_,x[i+"year"] = date_to_cat(x)
            x = x.drop(i,axis = 1)
            cat_variable = cat_variable + [i+"month",i+"year"]
            
            x_test[i+"month"],_,x_test[i+"year"] = date_to_cat(x_test)
            x_test = x_test.drop(i,axis = 1)
    
    num_variable = x.drop(cat_variable,axis = 1).columns.to_list()
    
    enc = OrdinalEncoder()
        
    # encode categorical variable 
    X_cat = x.loc[:,cat_variable]    
    enc.fit(X_cat)
    x.loc[:,cat_variable] = enc.transform(X_cat)
    
    X_cat_test = x_test.loc[:,cat_variable] 
    x_test.loc[:,cat_variable] = enc.transform(X_cat_test)
        
    if scale == True:
        # scale numerical variable
        X_num = x.loc[:,num_variable]
        scaler = MinMaxScaler()
        x.loc[:,num_variable] = scaler.fit_transform(X_num)
        
        X_num_test = x_test.loc[:,num_variable]
        scaler_test = MinMaxScaler()
        x_test.loc[:,num_variable] = scaler_test.fit_transform(X_num_test)
    return x, x_test


# model
def model(x,y):
    eclf = RandomForestClassifier(criterion = 'gini', max_depth = 75)
    eclf.fit(x,y)
    return eclf

# model fitting
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


# save processed test data
x_test1.to_csv('processed_test.csv',index=False)

#fit and save the fitted model
clf = model(x_train1,y_train)
pickle.dump(clf, open('model.pkl','wb'))
