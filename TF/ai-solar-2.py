import AIHelp

import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set()

filename = str(sys.argv[1])
df = pd.read_csv(filename, index_col='Time')

def select_columns(df):
	cols_to_keep = ['power_control_unit_cm_a_curr', 'power_control_unit_cm_a_volt', 'solar_array_misalignment']
	df_subset = df[cols_to_keep]
	return df_subset
	
def onehot_encode_integers(df, excluded_cols):
	df = df.copy()
	
	int_cols = [col for col in df.select_dtypes(include=['int']) if col not in excluded_cols]
	
	df.loc[:, int_cols] = df.loc[:, int_cols].astype('str')
	
	df_encoded = pd.get_dummies(df)
	
	return df_encoded
	
def normalize_cnt(df):
	df = df.copy()
	df['power_control_unit_cm_a_curr'] = df['power_control_unit_cm_a_curr'] / df['power_control_unit_cm_a_curr'].max()
	return df
	
dataset = (df.pipe(select_columns).pipe(onehot_encode_integers, excluded_cols=['power_control_unit_cm_a_curr']).pipe(normalize_cnt))

def create_dataset(df, n_deterministic_features, window_size, forecast_size, batch_size):
	shuffle_buffer_size = len(df)
	
	total_size = window_size + forecast_size
	
	data = tf.data.Dataset.from_tensor_slices(df.values)
	
	data = data.window(total_size, shift=1, drop_remainder=True)
	data = data.flat_map(lambda k: k.batch(total_size))
	
	data = data.shuffle(shuffle_buffer_size, seed=42)
	
	data = data.map(lambda k: ((k[:-forecast_size], k[-forecast_size:, -n_deterministic_features:]),
		k[-forecast_size:, 0]))
		
	return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
	
val_time = int(len(df)*0.8)
test_time = int(len(df)*0.9)

window_len = 24 * 3
forecast_len = 24

n_total_features = len(dataset.columns)
n_aleatoric_features = len(['power_control_unit_cm_a_curr', 'power_control_unit_cm_a_volt'])
n_deterministic_features = n_total_features - n_aleatoric_features

train_data = dataset.iloc[:val_time]
val_data = dataset.iloc[val_time:test_time]
test_data = dataset.iloc[test_time:]

batch_size = 32

train_window = create_dataset(train_data, n_deterministic_features, window_len, forecast_len, batch_size)
val_window = create_dataset(val_data, n_deterministic_features, window_len, forecast_len, batch_size)
test_window = create_dataset(test_data, n_deterministic_features, window_len, forecast_len, batch_size=1)

latent_dim = 72#int(sys.argv[2])

tfd = tfp.distributions

past_inputs = tf.keras.Input(shape=(window_len, n_total_features), name='past_inputs')
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(past_inputs)

future_inputs = tf.keras.Input(shape=(forecast_len, n_deterministic_features), name='future_input')
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])

x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
x = tf.keras.layers.Dense(int(latent_dim/2), activation='relu')(x)
#x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(2, activation='relu')(x)
#output = tf.keras.layers.Dense(1, activation='relu')(x)
output = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[...,0], scale=0.01*tf.math.softplus(t[...,1])), name='normal_dist')(x)

model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)

def neglogik(y, p_y):
	return -p_y.log_prob(y)

optimizer = tf.keras.optimizers.Adam(lr=0.003)
#loss = tf.keras.losses.Huber()
model.compile(loss=neglogik, optimizer=optimizer, metrics=["mae"])

history = model.fit(train_window, epochs=30, validation_data=val_window)

print(model.summary())

model.evaluate(test_window)

scaling_factor = df.power_control_unit_cm_a_curr.max()
mean = lambda x: x.mean().numpy().flatten() * scaling_factor
sd = lambda x: x.stddev().numpy().flatten() * scaling_factor

t = np.array(range(forecast_len))

AIHelp.plot_learning(filename + " - Solar Single Shot", history)

for i, data in enumerate(test_window.take(25)):
	### Plot stuff
	plt.clf()
	(past, future), truth = data
	truth = truth * scaling_factor

	pred = model((past,future))

	plt.fill_between(t, 
		mean(pred) + 2*sd(pred), 
	(mean(pred) - 2*sd(pred)).clip(min=0),
	color = 'green', label = '95% CI', alpha=0.2,
	linewidth = 2)
	plt.plot(t,mean(pred), label='Prediction', color='lightcoral', linewidth = 2)
	plt.plot(t,truth.numpy().flatten(),label='True value', linewidth = 2)

	plt.legend()

	plt.savefig(filename.split('.')[0] + '_interval_forecasts_' + str(i) + '.png')
	
	### Save to csv
	df = pd.DataFrame({"Time" : t, "Truth" : truth.numpy().flatten(), "Pred" : mean(pred), 
		"Sigp" : mean(pred) + 2*sd(pred), "Sign" : (mean(pred) - 2*sd(pred)).clip(min=0)})
	df.to_csv(filename.split('.')[0] + '_interval_forecast_' + str(i) + '.csv')
