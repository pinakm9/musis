from utility import *
import scipy.io.wavfile as sw
import python_speech_features as sf
import numpy as np
import pandas as pd
import os
from matplotlib.mlab import PCA

#@timer
def mfcc(file = "./../data/genres_wav/metal/metal.00092.wav"):
	samplerate, signal = sw.read(file)
	chunks = np.array_split(sf.mfcc(signal, samplerate, nfft = 1024), 4)
	features = []
	for chunk in chunks:
		features.append(chunk[:fpf/4])
	return np.array(features).flatten()

def feature_names(char = 'f', n = fpf*13):
	names = []
	for i in range(n):
		names.append(char + str(i))
	return names

def select(tfrac = 0.6, random = False):
	train, test = [], []
	for folder in ls_full_path(p2_wav):
		l = ls_full_path(folder)
		if random == True:
			np.random.shuffle(l)
		for file in l[:int(100*tfrac)]:
			train.append(file)
		for file in l[int(100*tfrac):]:
			test.append(file)
	return [train, test]

def label(file):
	y = [0.0 for i in range(10)]
	for i, genre in enumerate(genres):
		if genre in file:
			y[i] = 1.0
	return np.array(y)

@timer
def store(tfrac = 0.7, random = False):
	train, test = select(tfrac, random)
	d, l= {}, {}
	features, labels = feature_names(), feature_names('g', 10)
	for i, file in enumerate(train):
		d[i] = pd.Series(mfcc(file), index = features)
		l[i] = pd.Series(label(file), index = labels)
	df = pd.DataFrame(d)
	df.to_csv(p2_train, sep ='\t')
	df = pd.DataFrame(l)
	df.to_csv(p2_train_label, sep ='\t')
	d, l = {}, {}
	for i, file in enumerate(test):
		d[i] = pd.Series(mfcc(file), index = features)
		l[i] = pd.Series(label(file), index = labels)
	df = pd.DataFrame(d)
	df.to_csv(p2_test, sep ='\t')
	df = pd.DataFrame(l)
	df.to_csv(p2_test_label, sep ='\t')
	with open(p2_train_list, 'w') as file:
		for f in train:
			file.write(f.split('/')[-1] + '\n')
	with open(p2_test_list, 'w') as file:
		for f in test:
			file.write(f.split('/')[-1] + '\n')

class Datum:
	def __init__(self):
		self.music = []
		self.labels = []
		self.cur_pos = 0
	
	def next(self, batch):
		end = self.cur_pos + batch
		music = self.music[self.cur_pos: end]
		labels = self.labels[self.cur_pos: end]
		if end < len(self.music):
			self.cur_pos = end
		else:
			self.cur_pos = 0
		return [music, labels]


class MusicDB:

	def __init__(self, trm, trl, tem, tel):
		self.train = Datum()
		self.test = Datum()
		self.train.music = self.extract(trm)
		self.train.labels = self.extract(trl)
		self.test.music = self.extract(tem)
		self.test.labels = self.extract(tel)

	@timer	
	def extract(self, file, dtype = 'float32'):
		df = pd.read_csv(file, sep = '\t')
		return np.array([list(x)[1:] for x in df.T.itertuples()][1:], dtype = dtype)

	@timer
	def remove_genre(self, genre):
		l = label(genre)
		j = np.nonzero(l==1)[0][0]
		d = []
		for i, item in enumerate(self.train.labels):
			if item[j] == 1:
				d.append(i)
		self.train.music = np.delete(self.train.music, d, axis = 0)
		self.train.labels = np.delete(self.train.labels, d, axis = 0)
		d = []
		for i, item in enumerate(self.test.labels):
			if item[j] == 1:
				d.append(i)
		self.test.music = np.delete(self.test.music, d, axis = 0)
		self.test.labels = np.delete(self.test.labels, d, axis = 0)

	@timer
	def remove_genres(self, genres_to_remove):
		f = lambda x: np.nonzero(label(x) == 1)[0][0]
		d = list(map(f, genres_to_remove))
		for genre in genres_to_remove:
			self.remove_genre(genre)
		labels = []
		for item in self.train.labels:
			labels.append(np.delete(item, d, axis = 0))
		self.train.labels = np.array(labels, dtype = 'float32')
		labels = [] 
		for item in self.test.labels:
			labels.append(np.delete(item, d, axis = 0))
		self.test.labels = np.array(labels, dtype = 'float32')

#store(tfrac = 0.5, random = True)