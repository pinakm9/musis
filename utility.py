from time import time
import os

fpf = 160 # features per frequency (choose as a multiple of 4)

# Path variables used by different modules 
p2_data = './../data/genres/'
p2_wav = './../data/genres_wav/'
p2_train = './../data/train.csv'
p2_test = './../data/test.csv'
p2_train_label = './../data/train_label.csv'
p2_test_label = './../data/test_label.csv'
p2_train_list = './../data/train_list.txt'
p2_test_list = './../data/test_list.txt'
p2_m2 = 'models/model2/model2.ckpt'
p2_m3 = 'models/model3/model3.ckpt'
p2_results = 'results.txt'

# Genres in our database and colors for plotting
genres =  ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
colors = ['blue', 'green', 'magenta', 'yellow', 'black', 'red', 'darkgrey', 'lightcoral', 'indigo', 'maroon']

# Timing wrapper
def timer(func):
	def new_func(*args,**kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by function {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

# Similar ls command ubuntu, lists subdirectories 1 level deep
def ls_full_path(d):
    l = [os.path.join(d, f) for f in os.listdir(d)]
    l.sort()
    return l

# Creates directory if it doesn't exist
def mkdir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

# Identifies genres from their positions in genres (represented by a numeric string e.g. '075')
# If rest is True returns a list containing the genres not in the input list
def gmap(string, rest = False):
	l = []
	pos = list( map(int, list(string)) )
	if rest is False:
		for p in pos:
			l.append(genres[p])
	else:
		pos_ = list(range(len(genres)))
		for p in pos:
			pos_.remove(p)
		for p in pos_:
			l.append(genres[p]) 
	return l

# Returns remaining genres in genres, input is a list of strings
def remaining(genres_to_remove):
	l = [g for g in genres]
	for genre in genres_to_remove:
		l.remove(genre)
	return l