from mylinearregression import MyLinearRegression as MyLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("spacecraft_data.csv")

meters = np.array(data["Terameters"]).reshape((-1, 1))
price = np.array(data["Sell_price"]).reshape((-1, 1))

meters_norm = (meters - meters.mean()) / (max(meters) - min(meters))
price_norm = (price - price.mean()) / (max(price) - min(price))

mylr_meters = MyLR([0.0, 0.0], max_iter = 100000)
mylr_meters.fit_(meters_norm, price_norm)
price_norm_hat = mylr_meters.predict_(meters_norm)

price_hat = price_norm_hat * (max(price) - min(price)) + price.mean()

print("cost: {}".format(mylr_meters.mse_(price, price_hat)))

plt.xlabel("x3: distance totalizer value of spacecraft (in Tmeters)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(meters, price, color = "darkmagenta", s = 40, label = "Sell price")
plt.scatter(meters, price_hat, color = "hotpink", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()
