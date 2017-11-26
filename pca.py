import numpy as np
from matplotlib.mlab import PCA
import process as pr
from utility import *
from matplotlib import pyplot as plt
import matplotlib

def remaining(genres_to_remove)
	l = genres
	for genre in genres_to_remove:
		l.remove(genre)
	return l

def pca_plot(axis1, axis2, genres_to_remove):
	colors = ['blue', 'green', 'magenta', 'yellow', 'black', 'red', 'fuchsia', 'lightcoral', 'indigo', 'maroon']
	# Process data and compute PCA
	gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)
	gtzan.remove_genres(genres_to_remove)
	mfcc_pca = PCA(gtzan.train.music.T)
	# Create color map
	genre = 10 - len(genres_to_remove)
	spg = mfcc_pca.Wt[0].shape[0]/genre
	color_map = []
	i = 0
	color = colors[i]
	while True:
		color_map.append(color)
		i += 1 
		if i == spg*genre:
			break
		if i%spg == 0:
			color = colors[i/spg]
	# Make sure plots folder exists
	mkdir('plots')
	# Plot
	X = mfcc_pca.Wt[axis1-1]
	Y = mfcc_pca.Wt[axis2-1]
	plt.scatter(X, Y, c = color_map)
	plt.xlabel('pca' + str(axis1))
	plt.ylabel('pca' + str(axis2))
	
	plt.savefig('plots/pca_' + str(axis1) + '_' + str(axis2) + '.png')

pca_plot(1, 2, ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', ])
