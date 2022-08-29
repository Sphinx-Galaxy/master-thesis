import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

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
		
	def kernel(self):
		return tfp.math.psd_kernels.ExponentiatedQuadratic(
			amplitude=tf.nn.softplus(0.1 * self._amplitude),
			length_scale=tf.nn.softplus(5. * self._length_scale))

tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')

num_inducing_points = 40
model = tf.keras.Sequential([
	tf.keras.layers.InputLayer(input_shape=[1], dtype=x.dtype),
	tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
	tfp.layers.VariationalGaussianProcess(
		num_inducing_points=num_inducing_points, 
		kernel_provider=RBFKernelFn(dtype=x.dtype), 
		event_shape=[1], 
		inducing_index_points_initializer=tf.constant_initializer(
			np.linspace(*x_range, num=num_inducing_points, dtype=x.dtype)[..., np.newaxis]),
		unconstrained_observation_noise_variance_initializer=(
			tf.constant_initializer(np.expm1(1.).astype(x.dtype))),
		)
	])

batch_size = 32
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.03), loss=lambda y, rv_y: -rv_y.variational_loss(y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0]))
model.fit(x, y, epochs=500,  verbose=False)

yhat = model(x_test)

plt.plot(x, y, 'b.')

for i in range(7):
	sample_ = yhat.sample().numpy()
	plt.plot(x_test, sample_[..., 0].T, 'r', linewidth=0.9)
	
plt.show()

