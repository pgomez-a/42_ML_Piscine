from mylinearregression import MyLinearRegression as MyLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("spacecraft_data.csv")

X = np.array(data["Age"]).reshape((-1, 1))
X = np.insert(X, 1, np.array(data["Thrust_power"]), 1)
X = np.insert(X, 1, np.array(data["Terameters"]), 1)
X = X.transpose()
X_norm = X
for pos in range(len(X)):
    X_norm[pos] = (X[pos] - X[pos].mean()) / (max(X[pos]) - min(X[pos]))
X_norm = X_norm.transpose()

price = np.array(data["Sell_price"]).reshape((-1, 1))
price_norm = (price - price.mean()) / (max(price) -  min(price))

mylr = MyLR([0.0, 0.0, 0.0, 0.0], max_iter = 100000)
mylr.fit_(X_norm, price_norm)
price_norm_hat = mylr.predict_(X_norm)

price_hat = price_norm_hat * (max(price) - min(price)) + price.mean()

print("cost: {}".format(mylr.mse_(price, price_hat)))


plt.xlabel("x1: age (in years)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X[0], price, color = "darkblue", s = 40, label = "Sell price")
plt.scatter(X[0], price_hat, color = "dodgerblue", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()

plt.xlabel("x2: thrust power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X[1], price, color = "green", s = 40, label = "Sell price")
plt.scatter(X[1], price_hat, color = "lime", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()

plt.xlabel("x3: distance totalizer value of spacecraft (in Tmeters)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(X[2], price, color = "darkmagenta", s = 40, label = "Sell price")
plt.scatter(X[2], price_hat, color = "hotpink", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()
