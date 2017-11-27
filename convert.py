# Converts GTZAN database to .wav format

import subprocess as sp
import os
from utility import *

p2_conv = './../../genres_wav/'

# Make sure genres_wav folder exists
mkdir(p2_wav)
# Convert .au files to .wav files
for folder in os.listdir(p2_data):
	directory = p2_wav + folder + '/'
	print('Working in {}'.format(directory))
	if not os.path.exists(directory):
		os.makedirs(directory)
	for file in os.listdir(p2_data + folder + '/'):
		command = ['sox', p2_data + folder + '/' + file, '-e', 'float', directory + file[:-2] + 'wav']
		sp.Popen(command, stdout = sp.PIPE)