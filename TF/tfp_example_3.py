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
		
class RBFKernelFn(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RBFKernelFn, self).__init__(**kwargs)
		dtype = kwargs.get('dtype', None)
		
		self._amplitude = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='amplitude')
			
		self._length_scale = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='length_scale')
			
	def call(self, x):
		return x
		
	@property
	def kernel(self):
		return tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=tf.nn.softplus(0.1 * self._amplitude),
			length_scale=tf.nn.softplus(5. * self._length_scale))

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')

num_inducing_points = 40
model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape=[1]),
	tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
	tfp.layers.VariationalGaussianProcess(
		num_inducing_points=num_inducing_points, 
		kernel_provider=RBFKernelFn(), 
		event_shape=[1], 
		inducing_index_points_initializer=tf.constant_initializer(
			np.linspace(*x_range, num=num_inducing_points, dtype=x.dtype)[..., np.newaxis]),
		unconstrained_observation_noise_variance_initializer=(
			tf.constant_initializer(np.array(0.54).astype(x.dtype))),
		)
	])

batch_size = 32
loss = lambda y, rv_y: rv_y.variational_loss(
	y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=loss)
model.fit(x, y, batch_size=batch_size, epochs=500,  verbose=False)

yhat = model(x_test)

y, x, _ = load_dataset()

NX = []
NY = []

plt.plot(x, y, 'b.')

for i in range(3):
	sample_ = yhat.sample().numpy()
	plt.plot(x_test, sample_[..., 0].T, 'r', linewidth=0.9)
	
	NX.append(np.array(x_test))
	NY.append(np.array(sample_[..., 0].T))
	
#plt.show()

print("<<<NX>>>")
print((np.array(NX)).shape)
print("<<<NY>>>")
print((np.array(NY)).shape)

df = pd.DataFrame({"X" : np.array(x).flatten(), "Y" : np.array(y).flatten(),
	"NX1" : np.array(NX[0]).flatten(), "NY1" : NY[0],
	"NX2" : np.array(NX[1]).flatten(), "NY2" : NY[1],
	"NX3" : np.array(NX[2]).flatten(), "NY3" : NY[2]})
df.to_csv("tfp_example_3.csv", mode="w", index=False)
