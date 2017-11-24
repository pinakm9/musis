from time import time
import os

fpf = 160 # features per frequency (choose as a multiple of 4)
p2_wav = './../data/genres_wav/'
p2_train = './../data/train.csv'
p2_test = './../data/test.csv'
p2_train_label = './../data/train_label.csv'
p2_test_label = './../data/test_label.csv'
p2_train_list = './../data/train_list.txt'
p2_test_list = './../data/test_list.txt'

# Timing wrapper
def timer(func):
	def new_func(*args,**kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print('Time taken by function {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

def ls_full_path(d):
    l = [os.path.join(d, f) for f in os.listdir(d)]
    l.sort()
    return l
