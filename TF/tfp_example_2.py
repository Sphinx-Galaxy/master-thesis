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

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
	n = kernel_size + bias_size
	c = np.log(np.expm1(1.))
	
	return tf.keras.Sequential([
		tfp.layers.VariableLayer(2 * n, dtype=dtype),
		tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1))
		])
		
def prior_trainable(kernel_size, bias_size=0, dtype=None):
	n = kernel_size + bias_size
	
	return tf.keras.Sequential([
		tfp.layers.VariableLayer(n, dtype=dtype),
		tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1))
		])
		
tfd = tfp.distributions

model = tf.keras.Sequential([
	tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable),
	tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=lambda y, p_y: -p_y.log_prob(y))	
model.fit(x, y, epochs=500,  verbose=False)

yhats = [model(x_test) for _ in range(10)]
NX = []
NY = []
avgm = np.zeros_like(x_test[..., 0])

plt.plot(x, y, 'b.')

for i, yhat in enumerate(yhats):
	m = np.squeeze(yhat.mean())
	s = np.squeeze(yhat.stddev())

	plt.plot(x_test, m)
	avgm +=m
	
	NX.append(x_test[0])
	NX.append(x_test[-1])
	
	NY.append(m[0])
	NY.append(m[-1])
	
	print(NX)
	print(NY)
	
plt.plot(x_test, avgm/len(yhats), 'r', linewidth=4)

print(avgm/len(yhats))

plt.show()

df_1 = pd.DataFrame({"X" : np.array(x).flatten(), "Y" : np.array(y).flatten()})
df_2 = pd.DataFrame({"NX" : np.array(NX).flatten(), "NY" : NY})

df_1.to_csv("tfp_example_2_1.csv", mode="w", index=False)
df_2.to_csv("tfp_example_2_2.csv", mode="w", index=False)
