###########################################################################################
# Title 	: Melatih data EEG dengan menggunakan Recurrent Neural Network
# Author	: Achmad M. Gani
# Aims  	: 1. Menerapkan metode neural network berbasis time series dengan program Python
#	          2. Meramal nilai data eeg untuk masa depan
# Input		: Dataset EEG format edf
# Output	: Menampilkan nilai statistik RMSE
# Outline Code	: 1. Header
#		          2. Definisi Fungsi
#		          3. Algorithma
#		          4. Prediksi / Forecasting
############################################################################################
#--------------- DAFTAR FUNGSI--------------------
"""
build_timeseries(mat)
difference(dataset, interval=1)
train_test_split(dataset, scaler)
split_sequence(sequence, n_steps):
trim_dataset(mat, batch_size):
get_statistik(raw, data, indeks)
total_energy(coeffs):
normalisasi(x_train, x_test, scaler):
load_data(pasien, file)
get_features(channel, len_segment, raw_values):
train_kejang(pasien, file, awal_anotasi, akhir_anotasi, lstm_model):
train_bukan(pasien,awal,akhir,lstm_model):
train_pasien1(lstm_model):
train_pasien2(lstm_model)
train_pasien3(lstm_model)
train_pasien4(lstm_model)
train_pasien5(lstm_model)
tes(lstm_model,scaler)
save_model(name,model)
save_scaler(name,scaler)
load_model_scaler(name1='model1_ch18', name2='scaler1_ch18')
build_model(BATCH_SIZE, TIME_STEPS, x_train, output_bias = None, METRICS = ['accuracy'])
auc_roc(y_true, y_pred)
data_dist(x_norm)
plot_metrics(history)
plot_loss(history, label, n)
get_features_mp(len_segment, channel, raw_values)
get_features2(ch, len_segment, raw_values)
get_statistik2(raw)
plot_cm(labels, predictions, p=0.5)
"""
#--------------- UPDATE LOG-----------------------
"""
lstm_eeg_1.h5
- training model chbmit01_01 ~ 18

tasks :
1. explore k cross validation
2. imbalance data cases
	- test weighted values
	- test undersampling
	- test oversampling
3. deploy multiprocessing python
"""
#-------------------------------------------------
import time
import mne
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

from numpy import array
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import skew
from scipy.stats import kurtosis
from keras.models import load_model
import statistics
import joblib as jb

#data distribution
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.use('Tkagg')
import matplotlib.pyplot as plt


#time-frequency
import pywt
from pywt import wavedec

#paralel
import concurrent.futures
#tensorboard
import tensorboard
from datetime import datetime
from packaging import version

# %load_ext tensorboard
#set color
starttime=time.time()
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#Set the parameter
BATCH_SIZE = 5 #batch_size 10 kalau dari tsiouris2018
TIME_STEPS = 6 #input size 10 sampe 40an kalau dari tsiouris 2018
#detik = 10 # window size dalam satuan waktu
#detik coba di 8
detik = 10
frek = 256 #frekuensi sampling CHBMIT
unit = (detik*frek)-1 # window sampling
initial_bias = np.log([7/360])

#parameter matplotlib
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#--------------- PREDEFINE FUNCTIONS--------------
def build_timeseries(mat):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0+1, TIME_STEPS, dim_1))

    for i in range(dim_0+1):
        x[i] = mat[i:TIME_STEPS+i]
    print("length of time-series", x.shape)
    return x

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return array(diff)

def train_test_split(dataset, scaler):
    train_size = int(len(dataset) * scaler)
    train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
    return train, test

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def trim_dataset(mat, batch_size):
    no_of_rows = mat.shape[0]% batch_size
    if(no_of_rows > 0) :
        return mat[:-no_of_rows]
    else:
        return mat

def get_statistik(raw, data, indeks):
	data[indeks, 0] = raw.mean()
	data[indeks, 1] = statistics.variance(raw)
	data[indeks, 2] = raw.max() #skew raw
	data[indeks, 3] = raw.min() # kurtosis
	return data

#deprecated function into dwt_features()
def total_energy(coeffs):
	total = 0
	for i in range(len(coeffs)):
		total += abs(coeffs[i]**2)
	return total

def nstd(coeffs):
	return coeffs.std()/(coeffs.max()-coeffs.min())

def get_dwt_features(coeffs, data, exclude = 2): # buang d1 dan d2 karena noise
	for level in range(1, 8 - exclude):
		data.append(total_energy(coeffs[level]))
		data.append(coeffs[level].max())
		data.append(coeffs[level].min())
		data.append(skew(coeffs[level]))
		data.append(kurtosis(coeffs[level]))
		data.append(nstd(coeffs[level]))
		#data[indeks, 5] = nstd
		#nenergy
	return data

def normalisasi(x_train, x_test, scaler):
	# Normalisasi data fitur EEG
	train = scaler.transform(x_train)
	test = scaler.transform(x_test)
	return train, test

def eegfile(pasien,file):
	# load the data
	if pasien < 10:
		if file < 10:
			EEGFILE = "D:\EEG\Dataset\chb0"+ str(pasien)+ "\chb0"+ str(pasien)+ "_0" + str(file) + ".edf"
		else:
			EEGFILE = "D:\EEG\Dataset\chb0"+ str(pasien)+ "\chb0"+ str(pasien)+ "_" + str(file) + ".edf"
	else:
		if file < 10:
			EEGFILE = "D:\EEG\Dataset\chb"+ str(pasien)+ "\chb"+ str(pasien)+ "_0" + str(file) + ".edf"
		else:
			EEGFILE = "D:\EEG\Dataset\chb"+ str(pasien)+ "\chb"+ str(pasien)+ "_" + str(file) + ".edf"
	print(EEGFILE)
	return EEGFILE

def drop_channel(raw,pasien,file):
	# channel yang diharapkan ada dalam model
	# FP1-F7, F7-T7,T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8, P8-O2,
	# FZ-CZ and CZ-PZ
	# sisanya dibuang
	#anomali data untuk setiap file
	if pasien != 13 or file != 40 :
		raw = raw.drop_channels(['P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1'])
	if pasien == 4 and (file == 8 or file == 28 or file == 11):
		raw = raw.drop_channels(['ECG'])
	elif pasien == 9:
		raw = raw.drop_channels(['VNS'])
	elif pasien >=11 and pasien !=15 and pasien != 23 and pasien !=24:
		if pasien != 13 or file != 40:
			if pasien != 18 and pasien !=20:
				raw = raw.drop_channels(['--0', '--1', '--2', '--3', '--4'])
			if pasien == 18 or pasien ==20:
				raw = raw.drop_channels(['.-0', '.-1', '.-2', '.-3', '.-4'])
			if pasien == 12 and file >= 33:
				raw = raw.drop_channels(['LOC-ROC'])
		elif pasien == 13 and file == 40:
			raw = raw.drop_channels(['--0', '--1', '--2', '--3'])
	elif pasien == 15:
		raw = raw.drop_channels(['PZ-OZ'])
		raw = raw.drop_channels(['--0', '--1', '--2', '--3', '--4'])
		raw = raw.drop_channels(['--5', 'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref', 'CP1-Ref', 'CP2-Ref', 'CP5-Ref',
								 'CP6-Ref'])

	return raw

def filter_noise(mat):
	# band-pass filtering in the range 1 Hz - 50 Hz
	mat.filter(0.5, 60., fir_design='firwin')
	return mat

def load_data(pasien, file):

	EEGFILE = eegfile(pasien,file)
	# Raw unprocessed data for comparison purpose
	raw = mne.io.read_raw_edf(EEGFILE, stim_channel='auto', preload=True)
	# ------PRECPROCESSING--------

	# Remove inconsistent channels
	##Data dari chb01_03.edf
	raw = drop_channel(raw,pasien,file)
	#filter FIR
	raw = filter_noise(raw)
	return raw.get_data()

def get_features(channel, len_segment, raw_values):
	start_time = time.time()
	# feature extraction domain waktu
	for ch in range(channel):
		temp = np.empty((len_segment, 4))  # inisiasi fitur waktu, jumlah fitur waktu ada 4
		temp_wave = np.empty((len_segment, 5))  # inisiasi fitur spektral (total energy)
		for i in range(len_segment):  # i adalah baris di feature vector
			temp_dwtstats = np.empty((len_segment, 5))  # inisiasi fitur spektral tambahan (min, max, std dll)
			raw_segment = raw_values[ch, 0 + (unit * i):unit + unit * i]
			# fitur_temporal
			temp = get_statistik(raw_segment, temp, i)

			#fitur spektral
			coeffs = wavedec(raw_segment, 'coif3', level=7)
			exclude = 2  # buang d1 dan d2 karena noise
			for level in range(1, 8 - exclude):
				temp_wave[i, level - 1] = total_energy(coeffs[level])
				temp_dwtstats = get_dwt_features(coeffs[level], temp_dwtstats, i)
				if level == 1:
					temp_stats = temp_dwtstats
				else:
					temp_stats = np.hstack([temp_stats, temp_dwtstats])
		temp_wave = np.hstack([temp_wave, temp_stats])
		if ch == 0:  # inisiasi matriks fitur sebelum di konkatenasi
			fitur_waktu = temp
			fitur_spektral = temp_wave
			print("stacking channel ", ch+1, " selesai")
		else:
			fitur_waktu = np.hstack([fitur_waktu, temp])  # stack horizontal fitur waktu
			fitur_spektral = np.hstack([fitur_spektral, temp_wave])
			print("stacking channel ", ch + 1, " selesai")

	print("Ekstraksi fitur temporal dan spektral selesai")
	fitur_total = np.hstack([fitur_waktu, fitur_spektral])
	print("--- %s seconds ---" % (time.time() - start_time))
	return fitur_total

def train_kejang(pasien, file, awal_anotasi, akhir_anotasi, lstm_model):
	weight_0 = 1 / 35300 * 35307 / 2
	weight_1 = 1 / 7 * 35307 / 2
	class_weight = {0: weight_0, 1: weight_1}
	"""
		#-------------------------Annotation---------------------------
		#output seizure chb01
		#360*(channel-1) cheatsheet untuk 10 detik 256 Hz window sampling (360)
		ch3 = 3
		ch4 = 4
		ch15 = 15
		ch16 = 16
		ch18 = 18
		ch21 = 21
		ch26 = 26

		for i in range(TIME_STEPS+360*(ch3-1)+299, TIME_STEPS+360*(ch3-1)+305):
			y[i] = 1

		for i in range(TIME_STEPS+360*(ch4-1)+146, TIME_STEPS+360*(ch4-1)+150):
			y[i] = 1


		for i in range(TIME_STEPS+360 * (ch15 - 1) + 173, TIME_STEPS+360*(ch15-1) + 178):
			y[i] = 1

		for i in range(TIME_STEPS+360*(ch16-1)+101, TIME_STEPS+360*(ch16-1)+107):
			y[i] = 1

		for i in range(TIME_STEPS+360*(ch18-1)+172,TIME_STEPS+360*(ch18-1)+181):
			y[i] = 1

		for i in range(TIME_STEPS+360 * (ch21 - 1) + 32, TIME_STEPS+360*(ch21-1) + 42):
			y[i] = 1
		for i in range(TIME_STEPS+360 * (ch26 - 1) + 186, TIME_STEPS+360*(ch26-1) + 196):
			y[i] = 1
			#-------------------------Annotation---------------------------
		"""
	# load the data
	uji_values = load_data(pasien, file)
	# segmentasi data
	len_segment = uji_values.shape[1] // (detik * frek)  # indeks segmentasi
	print("Total segmentasi adalah", len_segment)
	channel = uji_values.shape[0]

	# ekstrak fitur
	fitur_total = get_features(channel, len_segment, uji_values)

	# create the output classification
	y_uji = np.zeros((len(fitur_total), 1))

	# supervised learning
	for i in range((TIME_STEPS + awal_anotasi)//detik, (TIME_STEPS + akhir_anotasi+1)//detik):
		y_uji[i] = 1

	# normalization
	x_uji = scaler.transform(fitur_total)
	x_uji_lstm = build_timeseries(x_uji)
	x_uji_lstm = trim_dataset(x_uji_lstm, BATCH_SIZE)
	y_uji_lstm = y_uji[0:len(x_uji_lstm)]  # agar jumlah baris output sama dengan input

	# evaluasi data lain
	lstm_model.fit(x_uji_lstm, y_uji_lstm, epochs=100, batch_size=BATCH_SIZE) #class_weight=class_weight)
	return lstm_model

def train_bukan(pasien,awal,akhir,lstm_model):
	for file in range(awal,akhir+1):
		# load the data
		uji_values = load_data(pasien, file)
		# segmentasi data
		len_segment = uji_values.shape[1] // (detik * frek)  # indeks segmentasi
		print("Total segmentasi adalah", len_segment)
		channel = uji_values.shape[0]

		# ekstrak fitur
		fitur_total = get_features2(channel, len_segment, uji_values)
		fitur_total = np.array(fitur_total)
		# create the output classification
		y_uji = np.zeros((len(fitur_total), 1))

		# normalization
		x_uji = scaler.transform(fitur_total)
		x_uji_lstm = build_timeseries(x_uji)
		x_uji_lstm = trim_dataset(x_uji_lstm, BATCH_SIZE)
		y_uji_lstm = y_uji[0:len(x_uji_lstm)]  # agar jumlah baris output sama dengan input

		# update data lain
		lstm_model.fit(x_uji_lstm, y_uji_lstm, epochs=50, batch_size=BATCH_SIZE, shuffle = True)

	return lstm_model

def tes_lama(lstm_model,scaler, threshold_data='a', mode = 'evaluate'):
	#UJI COBA PASIEN 1, 2 dan 14
	for pasien in range(3):
		# load the data
		if pasien == 0:
			uji_values = load_data(1, 26)
		elif pasien == 1:
			uji_values = load_data(2, 16)
		elif pasien == 2:
			uji_values = load_data(14, 11)
		# segmentasi data
		len_segment = uji_values.shape[1] // (detik * frek)  # indeks segmentasi
		print("Total segmentasi adalah", len_segment)
		channel = uji_values.shape[0]
		fitur_total = get_features2(channel, len_segment, uji_values)

		#problem disini
		# feture selection for better performance
		#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
		#fitur_total = sel.fit_transform(threshold_data)

		# create the output classification
		y_uji = np.zeros((len(fitur_total), 1))

		# uji coba ch26
		if pasien == 0:
			for i in range(TIME_STEPS + (1862//detik), TIME_STEPS + (1963//detik)+1):
				y_uji[i] = 1
			awal = 1862//detik
			akhir = 1963//detik
			uji = (akhir-awal)//BATCH_SIZE
		elif pasien == 1:
			for i in range(TIME_STEPS + (130//detik), TIME_STEPS + (212//detik)+1):
				y_uji[i] = 1
			awal = 130//detik
			akhir = 212//detik
			uji = (akhir - awal) // BATCH_SIZE
		elif pasien == 2:
			for i in range(TIME_STEPS + (1838//detik), TIME_STEPS + (1879//detik)+1):
				y_uji[i] = 1
			awal = 1838//detik
			akhir = 1879//detik
			uji = (akhir - awal) // BATCH_SIZE
		# normalization
		fitur_total = np.array(fitur_total)
		x_uji_lstm = build_timeseries(fitur_total)
		#reshape data
		num_instances, num_time_steps, num_features = x_uji_lstm.shape
		x_uji_lstm = np.reshape(x_uji_lstm, (-1, num_features))
		x_uji_lstm = scaler.transform(x_uji_lstm)
		# reshape balik ke bentuk asal
		x_uji_lstm = np.reshape(x_uji_lstm, (num_instances, num_time_steps, num_features))

		x_uji_lstm = trim_dataset(x_uji_lstm, BATCH_SIZE)
		y_uji_lstm = y_uji[0:len(x_uji_lstm)]  # agar jumlah baris output sama dengan input

		# evaluasi data lain
		# history = lstm_model.fit(x_uji_lstm, y_uji_lstm, epochs=50, batch_size=BATCH_SIZE)
		if mode == 'evaluate':
			scores = lstm_model.evaluate(x_uji_lstm, y_uji_lstm, verbose=2, batch_size=BATCH_SIZE)
		elif mode == 'predict': #data pasien pertama
			y_predictions = lstm_model.predict(x_uji_lstm, batch_size=BATCH_SIZE)
			return y_predictions, y_uji_lstm
		elif mode =='fitpredict':
			history = lstm_model.fit(x_uji_lstm, y_uji_lstm,
									 epochs=100, batch_size=BATCH_SIZE,
									 validation_data=(trim_dataset(x_val, BATCH_SIZE),
													  trim_dataset(y_val, BATCH_SIZE)))  # class_weight=class_weight)
			y_predictions = lstm_model.predict(x_uji_lstm, batch_size=BATCH_SIZE)
			return y_predictions, y_uji_lstm

def save_model(name,model):
	model.save(str(name))

def save_scaler(name,scaler):
	jb.dump(scaler,name)

def load_model_scaler(name1='model1_ch18', name2='scaler1_ch18'):
	return load_model(name1), jb.load(name2)

#-----------On progress----------
METRICS = [
		keras.metrics.TruePositives(name='tp'),
		keras.metrics.FalsePositives(name='fp'),
		keras.metrics.TrueNegatives(name='tn'),
		keras.metrics.FalseNegatives(name='fn'),
		keras.metrics.BinaryAccuracy(name='accuracy'),
		keras.metrics.Precision(name='precision'),
		keras.metrics.Recall(name='recall'),
		keras.metrics.AUC(name='auc'),
	]

def build_model(BATCH_SIZE, TIME_STEPS, x_train, output_bias = None, METRICS = ['accuracy']):

	#bias karena imbalanced data
	if output_bias is not None:
		output_bias	= tf.keras.initializers.Constant(output_bias)
	# klasifikasi biner
	lstm_model = tf.keras.Sequential()
	lstm_model.add(
		layers.LSTM(128, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_train.shape[2]), dropout=0.2, recurrent_dropout=0.2,
			 stateful=False,
			 kernel_initializer='random_uniform',
			 return_sequences=True))
	lstm_model.add(
		layers.LSTM(128, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_train.shape[2]), dropout=0.2,
					recurrent_dropout=0.2,
					stateful=False,
					kernel_initializer='random_uniform'))
	lstm_model.add(
		layers.Dense(30, activation='relu'))  # relu activation with 30 memory units
	lstm_model.add(
		layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))  # sigmoid to create 0 or 1 predictions+
	# lstm_model.add(Dense(1, activation='tanh')) #tanh to create -1 or 1 predictions+
	# optimizer = optimizers.RMSprop(lr=0.00008)#opsi 1
	# optimizer = adam opsi 2
	opt = keras.optimizers.Adam(learning_rate=0.001)
	lstm_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=METRICS)
	print(lstm_model.summary())
	return lstm_model

def data_dist(x_norm):
	df_pos = pd.DataFrame(x_norm)
	df_neg = pd.DataFrame(x_norm)
	sns.jointplot(df_pos[1], df_pos[2], #cek max sama min dwt dari d7
				  kind='hex', xlim=(-0.5, 1), ylim=(-1, 1))
	plt.suptitle("Positive distribution")

	sns.jointplot(df_neg[1], df_neg[2],
				  kind='hex', xlim=(-0.5, 1), ylim=(-1, 1))
	_ = plt.suptitle("Negative distribution")

def plot_metrics(history):
	metrics = ['loss', 'auc', 'precision', 'recall']
	for n, metric in enumerate(metrics):
		name = metric.replace("_"," ").capitalize()
		plt.subplot(2,2,n+1)
		plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
		plt.plot(history.epoch, history.history['val_'+metric],
				color=colors[0], linestyle="--", label='Val')
		plt.xlabel('Epoch')
		plt.ylabel(name)
		if metric == 'loss':
			plt.ylim([0, plt.ylim()[1]])
		elif metric == 'auc':
			plt.ylim([0,1])
		else:
			plt.ylim([0,1])
		plt.legend()


def plot_loss(history, label, n):
	# Use a log scale to show the wide range of values.
	plt.semilogy(history.epoch, history.history['loss'],
				 color=colors[n], label='Train ' + label)
	plt.semilogy(history.epoch, history.history['val_loss'],
				 color=colors[n], label='Val ' + label,
				 linestyle="--")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	plt.legend()


def get_features_mp(len_segment, channel, raw_values):
	#start_time = time.time()
	fitur_total = list()
	channel_mp = [i for i in range(channel)]
	len_segment_mp = [len_segment for i in channel_mp]
	raw_values_mp = [raw_values for i in channel_mp]
	with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
		results = executor.map(get_features2, channel_mp, len_segment_mp, raw_values_mp)
	for result in results:
		fitur_total.append(result)
	#print("--- %s seconds ---" % (time.time() - start_time))
	return fitur_total

def ekstrak_dataset(pasien, file, raw):
	data_ekstrak = np.zeros((0, TIME_STEPS, fitur_total.shape[1]))
	awal = [w-detik for w in raw]
	akhir = [x+detik*6 for x in raw]

	for dataset in range(len(pasien)):
		raw_values = load_data(pasien[dataset], file[dataset])
		raw_ekstrak = raw_values[:,awal[dataset]*frek:akhir[dataset]*frek] #frek dalam kurung berarti + - 1


		# segmentasi data
		channel = raw_ekstrak.shape[0]
		len_segment = raw_ekstrak.shape[1] // (detik * frek)  # indeks segmentasi
		print("Total segmentasi adalah", len_segment)
		fitur = get_features2(channel, len_segment, raw_ekstrak)
		fitur = np.array(fitur)

		# supervised learning features
		x_bal_lstm = build_timeseries(fitur)
		data_ekstrak = np.vstack([data_ekstrak, x_bal_lstm])
	return data_ekstrak

def seizure_dataset():
	# load the data format (pasien, edf awal, edf akhir)

	pasien = [1,1,1,1,1,1,1,
			  2,2,
			  3,3,3,3,3,3,3,
			  4,4,4,4,4,4,4,
			  5,5,5,5,5,5,5,5,5,5,
			  6,6,6,6,6,6,6,6,6,6,
			  7,7,7,7,7,7,
			  8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
			  8, 8, 8, 8, 8,
			  9,9,9,9,9,9,
			  10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
			  11,11,11,11,11,11,11,11,11,11,11,11,11,
			  12,12,12,12,12,12,12,12,12,12,
			  12, 12, 12, 12, 12, 12,
			  12,12,12,
			  12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
			  13,13,13,13,13,13,13,13,13,13,
			  13, 13, 13,
			  15,15,15,15,15,15,15,15,15,15,15,
			  15,15,15,15 , 15,15,
			  16,16,16,16,16, 16,16,16,
			  17,17,17,17,17,17,
			  18,18,18,18,18 ,18,
			  19,19,19,19,19,19,
			  20,20,20,20,20,20,20,20,
			  21,21,21,21,21
			  ]

	file = [3,4,15,16,18,18,21,
			19,36,
			1,2,3,4,34,35,36,
			5,8,8,28,28,28,28,
			6,6,13,13,16,16,17,17,22,22,
			1,1,1,4,4,9,10,13,18,24,
			12,12,13,13,19,19,
			2, 2, 2, 5, 5, 5, 11, 11, 13, 13,
			13, 21, 21, 21, 21,
			6,8,8,8,8,19,
			12,20,20,27,30,31,31,38,38,89,
			82,92,99,99,99,99,99,99,99, 99, 99, 99,99,
			6,6,8, 8,8,8, 9, 9,10,10,
			11,23,23,23,23,23,
			33,33,36,
			38,38,38,38,38,42,42, 42,42,42,
			19,21,21,40,40,55 ,55,58,59, 60,
			62,62,62,
			6,6,10,15,15,15,17,20,22,22,22,
			28,28,28,31,31,40,
			10,11, 14, 16,17,17,17, 17,
			3,3,4,4,63,63,
			29,30,31,32, 35, 36,
			28,28,29,29 ,30,30,
			12,13,13,14,15,15,16,68,
			19,20,21,21,22
			]

	awal_raw = [2996, 1467, 1732, 1015, 1720, 1780, 327,
				3369, 2972,
				362,731,432,2162,1982,2592,1725,
				7804, 6446, 6516,1679,1749, 3782,3852,
				417,487, 1086,1156, 2317,2387, 2451,2521, 2348,2418,
				1724,7461,13525,327,6211,12500,10833,506,7799,9387,
				4920,4990,3285,3355,13688,13758,
				2670,2740,2810,2856,2926,2996,2988,3058,2417,2487,
				2557,2083,2153,2223,2293,
				12231, 2951,3020,9196,9266,5299,
				6313,6860,6930,2382,3021,3780,3850,4610,4680,1383,
				298,2695,1454,1524,1594,1664,1734,1804,1874,1944,2014,2084,2154,
				1665,3415,1426,1591,1957,2798,3082,3503, 593,811,
				1085,253,323 ,425, 495, 630,
				2185,2427,653,
				1548,2798,2966,3146,3364,699,945,1170,1676,2213,
				2077,934,1004,142,530,458,2436 ,2474,3339,638,
				851,1626,2664,
				272,342,1082,1591,1661,1731,1925, 607,760,830,900,
				876,946, 1016 ,1751,1821,834,
				2290,1120,1854 ,1214, 227, 1684, 2152,3280,
				2282,2352,3025,3095,3136,3206,
				3477,541,2087,1908, 2196,463,
				299,369,2964,3034, 3159,3229,
				94,1440,2498,1971,390,1689,2226,1393,
				1288, 2627, 2003,2073,2553
				]

	data_seizure = ekstrak_dataset(pasien, file, awal_raw)

	# load the data format (pasien, edf awal, edf akhir)
	pasien = [1,1,1,1,1,1,1,
			  2,2,
			  3,3,3,3,3,3,3,
			  3,3,3,3,3,3,3,
			  5,5,5,5,5,5,5,5,5,5,
			  6,6,6,6,6,6,6,6,6,6,
			  7,7,7,7,7,7,
			  8,8,8,8,8,8,8,8,8,8,
			  8,8,8,8,8,
			  9,9,9,9,9,9,
			  10,10,10,10,10,10,10,10,10,10,
			  11,11,11,11,11,11,11,11,11,11,11,11,11,
			  12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
			  12, 12, 12, 12, 12, 12,
			  12, 12, 12,
			  12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
			  13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
			  13, 13, 13,
			  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
			  15, 15, 15, 15, 15, 15,
			  16,16,16,16,16, 16,16,16,
			  17, 17, 17, 17, 17, 17,
			  18, 18, 18, 18, 18, 18,
			  19, 19, 19, 19, 19, 19,
			  20, 20, 20, 20, 20, 20, 20, 20,
			  21,21,21,21,21]



	file = [1,2,5,6,7,8,23,
			1,6,
			14,15,16,6,7,8,9,
			10,11,12,13,29,30,31,
			1,2,3,4,5,12,10,11,19,20,
			1,1,1,4,4,9,10,13,18,24,
			12,12,13,13,19,19,
			2,2,2,5,5,5,11,11,13,13,
			13,21,21,21,21,
			6,8,8,8,8,19,
			12,20,20,27,30,31,31,38,38,89,
			82,92,99,99,99,99,99,99,99, 99, 99, 99,99,
			6,6,8, 8,8,8, 9, 9,10,10,
			11,23,23,23,23,23,
			33,33,36,
			38, 38, 38, 38, 38, 42, 42, 42, 42, 42,
			19, 21, 21, 40, 40, 55, 55, 58, 59, 60,
			62, 62, 62,
			6, 6, 10, 15, 15, 15, 17, 20, 22, 22, 22,
			28, 28, 28, 31, 31, 40,
			10, 11, 14, 16, 17, 17, 17, 17,
			3, 3, 4, 4, 63, 63,
			29, 30, 31, 32, 35, 36,
			28, 28, 29, 29, 30, 30,
			12, 13, 13, 14, 15, 15, 16, 68,
			19,20,21,21,22]

	awal_raw = [2996, 1467, 1732, 1015, 1720, 1780, 327,
				3369, 2972,
				362,731,432,2162,1982,2592,1725,
				804, 446, 516,1679,1749, 1782,1852,
				417,487, 1086,1156, 2317,2387, 2451,2521, 2348,2418,
				1600,7000,13000,50,5900,11000,8500,750,2300,1500,
				2500,2600,3600,1500,10000,1000,
				2500,3000,1500,1000,1300,1500,2750,2000,2200,1200,
				1000,300,400,12,78,
				12150, 200,2900, 90,200,5000,
				6000,2000,3000,4000,1000,1000,2000,3200,2000,1000,
				500,2800,1000,1000,1200,1500,1600,1500,1000,200,500,1900,2000,
				1550,3500,1500,1200,1600,2400,2500,100,1200,200,
				1000,300,190,500,600,450,
				2100, 2200, 300,
				1000,1000, 500, 500, 600, 600, 700,1500,200,800,
				1077, 534, 14, 100, 600, 500, 2000, 1900, 3000, 690,
				1000, 1800, 300,
				202, 142, 1000, 1800, 661, 731, 925, 107, 160, 230, 500,
				1000, 600, 116, 151, 821, 84,
				290, 120, 154, 114, 323, 684, 152, 380,
				282, 252, 325, 395, 2136, 1206,
				1477, 741, 1087, 908, 196, 46,
				1299, 1369, 964, 334, 2359, 1229,
				200, 440, 498, 971, 500, 1500, 226, 393,
				288, 627, 203,273,553]

	data_non_seizure = ekstrak_dataset(pasien, file, awal_raw)

	y_seizure = np.ones((data_seizure.shape[0], 1))
	y_non_seizure = np.zeros((data_non_seizure.shape[0], 1))
	#gabungkan 2 data
	x_train_lstm = np.vstack([data_seizure, data_non_seizure])
	y_train_lstm = np.vstack([y_seizure, y_non_seizure])

	# apply normalisasi
	scaler = MinMaxScaler(feature_range=(0, 1))
	# reshape data and normalisasi
	num_instances, num_time_steps, num_features = x_train_lstm.shape
	x_train_lstm = np.reshape(x_train_lstm, (-1, num_features))
	x_train_lstm= scaler.fit_transform(x_train_lstm)

	#reshape balik ke bentuk asal
	x_train_lstm = np.reshape(x_train_lstm, (num_instances, num_time_steps, num_features))
	return x_train_lstm, y_train_lstm, scaler

def tes_baru():
	# load the data format (pasien, edf awal, edf akhir)

	pasien = [1,1,2,2,14,22,22,22,23,23,23,24,24,24]

	file = [26,26,16,16,11,20,25,38,9,9,9,3,3,4]

	awal_raw = [1862, 1932,130,200,1838,
				3377,3139,1280,2579,6885, 8505,231,2883,1411
				]

	data_seizure = ekstrak_dataset(pasien, file, awal_raw)

	# load the data format (pasien, edf awal, edf akhir)
	pasien = [1,1,2,2,3, 22,22,22,23,23,23,24,24,24]
	file = [26,1,7,8,10, 20,25,38,9,9,9,3,3,4]
	awal_raw = [10,300,300,500,200, 377,139,280,259,5885, 3505,531,1883,411]

	data_non_seizure = ekstrak_dataset(pasien, file, awal_raw)

	y_seizure = np.ones((data_seizure.shape[0], 1))
	y_non_seizure = np.zeros((data_non_seizure.shape[0], 1))
	#gabungkan 2 data
	x_train_lstm = np.vstack([data_seizure, data_non_seizure])
	y_train_lstm = np.vstack([y_seizure, y_non_seizure])

	# apply normalisasi
	# reshape data and normalisasi
	num_instances, num_time_steps, num_features = x_train_lstm.shape
	x_train_lstm = np.reshape(x_train_lstm, (-1, num_features))
	x_train_lstm= scaler.transform(x_train_lstm)

	#reshape balik ke bentuk asal
	x_train_lstm = np.reshape(x_train_lstm, (num_instances, num_time_steps, num_features))
	scores = lstm_model.evaluate(x_train_lstm, y_train_lstm, verbose=2, batch_size=BATCH_SIZE)

	#score masing2
	print("skor pasien prediksi : ")
	#y_predict = lstm_model.predict(x_train_lstm, batch_size=BATCH_SIZE)
	#y_predictions = lstm_model.predict(x_uji_lstm, batch_size=BATCH_SIZE)
	print(y_predict)
	return x_train_lstm

def plot_loss():
	from matplotlib import pyplot
	pyplot.plot(history.history['loss'])
	pyplot.plot(history.history['val_loss'])
	pyplot.title('model train vs validation loss')
	pyplot.ylabel('loss')
	pyplot.xlabel('epoch')
	pyplot.legend(['train', 'validation'], loc='upper right')
	pyplot.show()

def get_features2(ch, len_segment, raw_values):
	fitur_total = list()
	for len in range(len_segment):  # i adalah baris di feature vector
		temp = list()  # inisiasi fitur waktu, jumlah fitur waktu ada 4
		temp_wave = list()  # inisiasi fitur spektral
		for chan in range(ch):
			raw_segment = raw_values[chan, 0 + (unit * len):unit + unit * len]
			#fitur temporal
			temp.extend(get_statistik2(raw_segment))

			# fitur spektral
			coeffs = wavedec(raw_segment, 'coif3', level=7)
			exclude = 2  # buang d1 dan d2 karena noise
			temp_wave = get_dwt_features(coeffs, temp_wave)

		temp.extend(temp_wave)
		fitur_total.append(temp)
		if len % (len_segment//2) == 0:
			print("ekstraksi fitur ", round(len/len_segment*100), "%")
	return fitur_total

def get_features2_mp(ch, len_segment, raw_values):
	print("Sedang melakukan ekstraksi fitur channel ", ch)
	temp = list()# inisiasi fitur waktu, jumlah fitur waktu ada 4
	for i in range(len_segment):  # i adalah baris di feature vector
		temp_wave = list()  # inisiasi fitur spektral
		raw_segment = raw_values[ch, 0 + (unit * i):unit + unit * i]

		#fitur temporal
		temp.extend(get_statistik2(raw_segment))

		# fitur spektral
		coeffs = wavedec(raw_segment, 'db8', level=7)
		exclude = 2  # buang d1 dan d2 karena noise
		for level in range(1, 8 - exclude):
			temp_wave.append(total_energy(coeffs[level]))
		temp.extend(temp_wave)
	#fitur_total.append(temp)
	return temp

def get_statistik2(raw):
	data = []
	data.append(raw.mean())
	data.append(statistics.variance(raw))
	data.append(skew(raw))
	data.append(kurtosis(raw))
	data.append(raw.max())
	data.append(raw.min())
	return data

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))
# --------------- MAIN ALGORITHM-------------------

# model, scaler = load_model_scaler()

# load the data format (pasien, edf awal, edf akhir)
raw_values = load_data(1, 3)

# segmentasi data
len_segment = raw_values.shape[1] // (detik * frek)  # indeks segmentasi
print("Total segmentasi adalah", len_segment)

channel = raw_values.shape[0]
fitur_total= get_features2(channel, len_segment, raw_values)
fitur_total = np.array(fitur_total)
#feture selection for better performance
#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#fitur_total = sel.fit_transform(fitur_total_raw)


#train split test data
scale = 0.92
x_train = fitur_total[0:math.floor(scale*len(fitur_total)), :]
x_test = fitur_total[math.floor(scale*len(fitur_total)):len(fitur_total), :]


#create the output classification dan assign 0 bukan kejang
y = np.zeros((len(fitur_total), 1))


#data pasien 3
for i in range(TIME_STEPS+(2996//detik), TIME_STEPS+(3036//detik)+1):
	y[i] = 1

#split train test output data
y_train = y[0:math.floor(scale*len(fitur_total))]
y_test = y[math.floor(scale*len(fitur_total)):len(fitur_total)]

#insert oversampling seizure data
"""
x_seiz = np.zeros((310, fitur_total.shape[1]))
y_seiz = np.ones((310, 1))
k = 0
for m in range(x_seiz.shape[0]//5):
	for i in range(TIME_STEPS+(2996//detik), TIME_STEPS+(3036//detik)+1):
		x_seiz[k+i-(TIME_STEPS+(2996//detik))] = fitur_total[i]
	k = k + i-(TIME_STEPS+(2996//detik))


#insert the oversampling data into training data
x_train = np.vstack([x_train, x_seiz[:k]])
y_train = np.vstack([y_train, y_seiz[:k]])
"""
#Normalisasi data fitur EEG
#inisiasi scaler data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(x_train)

#insert seizure dataset
#x_seizure = np.load('seizure.npy')
x_seizure, y_seizure, scaler = seizure_dataset()
x_train_norm, x_test_norm = normalisasi(x_train, x_test, scaler)
#supervised learning features
x_train_lstm = build_timeseries(x_train_norm)
"""
#dataset2
data2 = dataset2()

#insert seizure dataset before trim
x_train_lstm = np.vstack([x_train_lstm, x_seizure, data2])
y_train_lstm = np.vstack([y_train, y_seizure, np.zeros((len(data2), 1))])
"""

#insert seizure dataset before trim
#x_train_lstm = np.vstack([x_train_lstm, x_seizure])
#y_train_lstm = np.vstack([y_train, y_seizure])


x_train_lstm = trim_dataset(x_seizure, BATCH_SIZE)
y_train_lstm = y_seizure[0:len(x_train_lstm)] #agar jumlah baris output sama dengan input

x_test= build_timeseries(x_test_norm)
#data validasi, data tes menggunakan fungsi tes(...)
x_val = trim_dataset(x_test, BATCH_SIZE)
y_val = y_test[0:len(x_test)] #agar jumlah baris output sama dengan input

#iseng
#counter = Counter(y_train_lstm)
#print(counter)

#building the over & under sample
"""over = SMOTE(sampling_strategy=0.2)
under = RandomUnderSampler(sampling_strategy=0.4)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

x_over, y_over = pipeline.fit_resample(x_train_lstm, y_train_lstm)

x_train_lstm = x_over
y_train_lstm = y_over
"""
#klasifikasi biner
#weighted value
weight_0 = 1/35300*35307/2
weight_1 = 1/7*35307/ 2
class_weight = {0: weight_0, 1: weight_1}
#end weighted value

# lstm_model = build_model(BATCH_SIZE, TIME_STEPS, x_train_lstm, METRICS=METRICS)
# # Define the Keras TensorBoard callback.
# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# history = lstm_model.fit(x_train_lstm, y_train_lstm,
# 						 epochs=500, batch_size=BATCH_SIZE,
# 						 validation_data=(trim_dataset(x_val, BATCH_SIZE),
#                    		 trim_dataset(y_val, BATCH_SIZE)), shuffle=True)#class_weight=class_weight)
# #modelfit
# #np.vstack([x_train_lstm, x_val]), np.vstack([y_train_lstm, y_val])
#
# #evaluation
# #scores = lstm_model.evaluate(x_test_lstm, y_test_lstm, verbose=0, batch_size=BATCH_SIZE)
# #print("Accuracy: %.2f%%" % (scores[1]*100)) #------------------------------UJI DATA LAIN
# #print("UJI DATA LAIN")
# print("--- %s seconds ---" % (time.time() - starttime))
