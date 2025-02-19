import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

def trainmodel():
    model =  LogisticRegression(fit_intercept=True,
            multi_class='auto', # indicating if binary classification or multiple categories
            penalty='l2', # cost function
            solver='saga', # machine learning model
            max_iter=10000, # training iterations
            C=50 #regularization
        )
    model.fit(x_train_scaled,y_train)
    # saving model pickle
    pickle.dump(model, open(filename, 'wb'))
    return model

def loadmodel():
    # loading the python object file
    model = pickle.load(open(filename, 'rb'))  
    return model

mnist = pd.read_csv("mnist_784.csv")
filename = 'mnist.model.sav'

X = mnist.drop("class",axis=1).head(550).values
Y = mnist.head(550)["class"].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test) 


LG = loadmodel()
pred = LG.predict(x_test_scaled)
print(confusion_matrix(y_test, pred))
print(print("Accuracy:",accuracy_score(y_test, pred)) )
