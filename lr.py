import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split



np.random.seed(0)

X = np.random.rand(100) *10 # 1000

noise = np.random.randn(100) * 1

y = 2.5 * X + 5 + noise# formula  y = m *x +b 


# linear regression 

# plt.scatter(X, y, color="blue", marker="o" )

# plt.title("linear regression")
# plt.grid(True)
# plt.show()
# X = X.reshape(-1 , 1)

def linearregression(X, y, learningrate=0.02, epochs=10000) : 
    m = 0
    c = 0 
    n= len(X)

    for  epoch  in range(epochs):
        # prediction 

        y_pred = m *X+ c

        # calculate errors 

        errors = y - y_pred 

        # gradient desent 

        dm= (-2/n) * np.sum(X * errors)
        dc =(-2/n) * np.sum(errors)

        #updating m and c gradient 

        m = m -learningrate * dm 
        c = c - learningrate * dc


        if epoch % 100 == 0: 
            mse = np.mean(errors ** 2)
            print(f"epoch {epoch} : Mse= {mse:4f}")

            return m, c 
        
m, c = linearregression(X, y)
print(f" Train model : y = {m:.2f}x + {c:.2f}")

# original data ko scatter plot karo
# plt.scatter(X, y, color='blue',label='Data points')

# # predicted line draw karo
# y_line = m * X * c

# plt.plot( X, y_line , color='red', label='Best Fit Line')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Linear Regression from Scratch')
# plt.legend()
# plt.show()

plt.scatter(X, y, label='Data Points')
y_line = m *  X+ c
sorted_idx =np.argsort(X)
plt.plot(X[sorted_idx], y_line[sorted_idx], color='red', label='Best Fit Line')
plt.legend()
plt.title("Linear Regression from scartch")
plt.show()