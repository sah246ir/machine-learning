import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_csv("Salaries.csv")

# Extracting all the features from which Y is to be predicted 
# in this case we are storing years of exp as feature
X = data.iloc[:,:-1].values 

# Extracting list of values that are to be predicted
# in this case we are storing salary
Y = data.iloc[:,1].values 

# splitting the data into train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


class LinearRegression():
    def __init__(self,learning_rate,iterations):
        # initializing learning rate and training iterations
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self,x_train,y_train):
        # initializing variables
        self.m,self.n = x_train.shape 
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = x_train
        self.Y = y_train
        
        # running iterations of gradient descent
        for i in range(self.iterations):
            self.gradient_descent()
        return self

    def predict(self,X):
        # implements estimator i.e. W.X + b
        return X.dot(self.W) + self.b

    def gradient_descent(self):
        # creating prediction based on current weight and bias.
        y_pred = self.predict(self.X)

        # calculating gradients or derivatives
        # derivatives will determine direction of flow on the gradient descent
        # first we calculate the current slope i.e. if we are moving towards the slope or away i.e. negative or positive slope
        # we reverse the slope which will change our direction towards the minima
        # we know that the gradient descent graph for linear regression is convex in shape
        # for ex we are on a negative slope of -2 then after reversing it becomes 2 which indicates we need to move towards right i.e. slope
        # if we are on a positive slope of 2 then after reversing it becomes -2 which indicates we need to move towards left i.e. slope
        # we transpose X to align the shape for multiplication with weight or the Y vector
        dw = -(self.X.T.dot(self.Y-y_pred) / self.m)
        db = -(np.sum(self.Y-y_pred) / self.m)

        # updating values of w and b simultaneously
        new_w = self.W-self.learning_rate*dw
        new_b = self.b-self.learning_rate*db
        self.W = new_w
        self.b = new_b
        
        return self 
    
lg = LinearRegression(0.02,1000)

lg.fit(X_train,y_train)

pred = lg.predict(X_test)

plt.scatter( X_test, y_test, color = 'blue' ) 
plt.plot( X_test, pred, color = 'red' ) 
plt.title( 'Salary vs Experience' ) 
plt.xlabel( 'Years of Experience' ) 
plt.ylabel( 'Salary' ) 
plt.savefig("linear-regression") 