import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_test = load_dataset()

tfd = tfp.distributions

model = tf.keras.Sequential([
	tf.keras.layers.Dense(1),
	tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=lambda y, p_y: -p_y.log_prob(y))	
model.fit(x, y, epochs=500,  verbose=False)

yhat = model(x_test)

mean = yhat.mean()

plt.plot(x, y, 'b.')
plt.plot(x_test, yhat.mean())
#plt.show()

df = pd.DataFrame({"X" : np.array(x).flatten(), "Y" : np.array(y).flatten(), "NX" : np.array(x_test).flatten(), "NY" : np.array(yhat.mean()).flatten()})
df.to_csv("tfp_example_1.csv", mode="w", index=False)

