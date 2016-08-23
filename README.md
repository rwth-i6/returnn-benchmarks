We are doing benchmarks.

### Speed benchmarks

In these experiments we used the following hardware:

1. Intel(R) Core(TM) i5-4690K CPU @ 3.50GHz

2. 16 Gb LDDR-3

3. Nvidia GTX-970

OS : Ubuntu 16.10 LTS, video driver : 367.35, CUDA 7.5, cuDNN-5 R5.

Network topology : 3 layer [B]LSTM with 512 hidden units per cell followed by softmax output layer.
All experiments available to reproduce using run scripts or corresponding config files.

| Framework        | Cell Type           | Avg. time per epoch  |
| ------------- |:-------------:| -----:|
| RETURNN    | LSTM | 320 sec |
| RETURNN    | BLSTM | 787 sec |
| Torch7 + cudnn | LSTM      |   156.5 sec|
| Torch7 + cudnn | BLSTM      |   wip |
| Keras [Theano] | LSTM  |   wip |
| Keras [Theano] | BLSTM  |   wip |
| TensorFlow     | LSTM   | wip |
| TensorFlow     | BLSTM  | wip |
