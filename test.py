import tensorflow as tf
import numpy as np
from utility import * 
import process as pr
# Data sets
gtzan = pr.MusicDB(p2_train, p2_train_label, p2_test, p2_test_label)
print gtzan.train.music.shape

gtzan.pcafy()

