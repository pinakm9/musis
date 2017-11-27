import numpy as np
from matplotlib.mlab import PCA
import process as pr
from utility import *
from matplotlib import pyplot as plt
import matplotlib
import itertools as it


@timer # Creates pca plot for training data, genres_to_keep is a numeric string
def pca_plot(axis1, axis2, genres_to_keep):
	# Process data and compute PCA
	gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)
	genres_to_remove = gmap(genres_to_keep, rest = True)
	gtzan.remove_genres(genres_to_remove)
	mfcc_pca = PCA(gtzan.train.music.T)
	genre = 10 - len(genres_to_remove)
	spg = mfcc_pca.Wt[0].shape[0]/genre
	# Make sure plots folder exists
	mkdir('plots')
	# Plot
	fig, ax = plt.subplots()
	rest = remaining(genres_to_remove)
	tag = ''
	for genre in rest:
		tag += str(genres.index(genre)) 
	for i, genre in enumerate(rest):
		color = colors[i]
		X = mfcc_pca.Wt[axis1-1][i*spg:(i+1)*spg]
		Y = mfcc_pca.Wt[axis2-1][i*spg:(i+1)*spg]
		plt.scatter(X, Y, c = color, label = genre)
	plt.xlabel('pca' + str(axis1))
	plt.ylabel('pca' + str(axis2))
	plt.legend()
	plt.savefig('plots/pca_' + str(axis1) + '_' + str(axis2) + '_' + tag + '.png')

@timer # Create all possible pca plots for groups with num_genre elements with axis1, axis2
# e.g. if num_genre = 3 then we'd get all possible pca plots for 3-genre groups
def pca_plt(axis1, axis2, num_genre):
	toStr = lambda comb: ''.join([str(x) for x in comb])
	for comb in it.combinations(list(range(len(genres))), num_genre):
		pca_plot(axis1, axis2, toStr(comb))

# Usage: creates a pca2 vs pca3 plot for the genres blues, classical, jazz
pca_plot(2, 3, '015')
