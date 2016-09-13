#!/usr/bin/env python

# inspired by https://github.com/fchollet/keras/blob/master/examples/imdb_bidirectional_lstm.py

from __future__ import print_function
import sys
import h5py
import numpy
numpy.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, TimeDistributedDense, Activation
from keras.utils import np_utils

import better_exchook
better_exchook.install()


class SimpleHdf:
	def __init__(self, filename):
		self.hdf = h5py.File(filename)
		self.seq_tag_to_idx = {name: i for (i, name) in enumerate(self.hdf["seqTags"])}
		self.num_seqs = len(self.hdf["seqTags"])
		assert self.num_seqs == len(self.seq_tag_to_idx), "not unique seq tags"
		seq_lens = self.hdf["seqLengths"]
		if len(seq_lens.shape) == 2: seq_lens = seq_lens[:, 0]
		self.seq_lens = seq_lens
		assert self.num_seqs == len(self.seq_lens)
		self.seq_starts = [0] + list(numpy.cumsum(self.seq_lens))
		total_len = self.seq_starts[-1]
		inputs_len = self.hdf["inputs"].shape[0]
		assert total_len == inputs_len, "time-dim does not match: %i vs %i" % (total_len, inputs_len)
		assert self.seq_starts[-1] == self.hdf["targets/data/classes"].shape[0]
		try:
			self.num_outputs = self.hdf.attrs['numLabels']
		except Exception:
			self.num_outputs = self.hdf['targets/size'].attrs['classes']

	def get_seq_tags(self):
		return self.hdf["seqTags"]

	def get_data(self, seq_idx):
		seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
		return self.hdf["inputs"][seq_t0:seq_t1]

	def get_targets(self, seq_idx):
		seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
		return self.hdf["targets/data/classes"][seq_t0:seq_t1]

	def get_data_dict(self, seq_idx):
		return {"data": self.get_data(seq_idx), "classes": self.get_targets(seq_idx)}

print('Loading data...')
data = SimpleHdf("data/train.0001")

print("Num sequences:", data.num_seqs)
seq_len = data.seq_lens[0]  # should all be the same
print("Sequence length:", seq_len)
input_dim, = data.hdf["inputs"].shape[1:]
print("Input data dimension:", input_dim)
output_dim = data.num_outputs
print("Output data number of labels:", output_dim)

X_train = numpy.array(data.hdf["inputs"], dtype="float32")
X_train = X_train.reshape((data.num_seqs, seq_len, input_dim))
assert numpy.array_equal(X_train[0], data.get_data(0))  # reshaping correct?

y_train = numpy.array(data.hdf["targets/data/classes"], dtype="int32")
y_train = y_train.reshape((data.num_seqs, seq_len))
assert numpy.array_equal(y_train[0], data.get_targets(0))  # reshaping correct?
y_train = numpy.where((0 <= y_train) * (y_train < output_dim), y_train, 0)  # fix some broken indices?
y_train = numpy.expand_dims(y_train, -1)  # needed for sparse_categorical_crossentropy


data = SimpleHdf("data/train.0001")
X_valid = numpy.array(data.hdf["inputs"], dtype="float32")
X_valid = X_train.reshape((data.num_seqs, seq_len, input_dim))
assert numpy.array_equal(X_valid[0], data.get_data(0))  # reshaping correct?

y_valid = numpy.array(data.hdf["targets/data/classes"], dtype="int32")
y_valid = y_valid.reshape((data.num_seqs, seq_len))
assert numpy.array_equal(y_valid[0], data.get_targets(0))  # reshaping correct?
y_valid = numpy.where((0 <= y_valid) * (y_valid < output_dim), y_valid, 0)  # fix some broken indices?
y_valid = numpy.expand_dims(y_valid, -1)  # needed for sparse_categorical_crossentropy

model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True), batch_input_shape=(None, None, input_dim)))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(TimeDistributed(Dense(output_dim=output_dim)))
model.add(TimeDistributed(Activation('softmax')))

# Note that sparse_categorical_crossentropy could be faster: https://github.com/fchollet/keras/issues/3649
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
print('Train...')
model.fit(
	X_train, y_train,
	batch_size=27,
	nb_epoch=1)

model.evaluate(X_valid,y_valid,batch_size=27)

