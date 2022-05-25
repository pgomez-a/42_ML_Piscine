from my_linear_regression import MyLinearRegression as MyLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(dataset["Micrograms"]).reshape(-1, 1)
Yscore = np.array(dataset["Score"]).reshape(-1, 1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

plt.title("Linear Model 1")
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.ylabel("Space driving score")
plt.scatter(Xpill, Yscore, color = "cyan", label = "True pills")
plt.scatter(Xpill, Y_model1, color = "lime", marker = "x", label = "True pills")
plt.plot(Xpill, Y_model1, color = "lime", linestyle = "dashed", label = "Predict pills")
plt.legend()
plt.show()

plt.title("Linear Model 2")
plt.xlabel("Quantity of blue pill (in micrograms)")
plt.ylabel("Space driving score")
plt.scatter(Xpill, Yscore, color = "cyan", label = "True pills")
plt.scatter(Xpill, Y_model2, color = "lime", marker = "x", label = "True pills")
plt.plot(Xpill, Y_model2, color = "lime", linestyle = "dashed", label = "Predict pills")
plt.legend()
plt.show()

print("mse of linear_model1: {}".format(MyLR.mse_(Yscore, Y_model1)))
print("mse of linear_model2: {}".format(MyLR.mse_(Yscore, Y_model2)))
