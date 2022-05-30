from mylinearregression import MyLinearRegression as MyLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("spacecraft_data.csv")

age = np.array(data["Age"]).reshape((-1, 1))
price = np.array(data["Sell_price"]).reshape((-1, 1))

mylr_age = MyLR([0.0, 0.0], max_iter = 100000)
mylr_age.fit_(age, price)
price_hat = mylr_age.predict_(age)

print("cost: {}".format(mylr_age.mse_(price, price_hat)))

plt.xlabel("x1: age (in years)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(age, price, color = "darkblue", s = 40, label = "Sell price")
plt.scatter(age, price_hat, color = "dodgerblue", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()
