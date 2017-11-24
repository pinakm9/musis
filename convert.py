import subprocess as sp
import os

p2_data = './../data/genres/'
p2_wav  = './../data/genres_wav/'
p2_conv = './../../genres_wav/'

for folder in os.listdir(p2_data):
	directory = p2_wav + folder + '/'
	print('Working in {}'.format(directory))
	if not os.path.exists(directory):
		os.makedirs(directory)
	for file in os.listdir(p2_data + folder + '/'):
		command = ['sox', p2_data + folder + '/' + file, '-e', 'float', directory + file[:-2] + 'wav']
		sp.Popen(command, stdout = sp.PIPE)