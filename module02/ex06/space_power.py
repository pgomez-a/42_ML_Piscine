from mylinearregression import MyLinearRegression as MyLR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("spacecraft_data.csv")

power = np.array(data["Thrust_power"]).reshape((-1, 1))
price = np.array(data["Sell_price"]).reshape((-1, 1))

power_norm = (power - power.mean()) / (max(power) - min(power))
price_norm = (price - price.mean()) / (max(price) - min(price))

mylr_power = MyLR([0.0, 0.0], max_iter = 100000)
mylr_power.fit_(power_norm, price_norm)
price_norm_hat = mylr_power.predict_(power_norm)

price_hat = price_norm_hat * (max(price) - min(price)) + price.mean()

print("cost: {}".format(mylr_power.mse_(price, price_hat)))

plt.xlabel("x2: thrust power (in 10Km/s)")
plt.ylabel("y: sell price (in keuros)")
plt.scatter(power, price, color = "green", s = 40, label = "Sell price")
plt.scatter(power, price_hat, color = "lime", s = 15, label = "Predicted sell price")
plt.grid(visible = True, alpha = 0.5)
plt.legend()
plt.show()
