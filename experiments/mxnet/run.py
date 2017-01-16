#!/usr/bin/env python
# pylint:skip-file
import sys
import h5py
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


BATCH_SIZE = 27
MAX_LEN = 250

class HDF5DATA(mx.io.DataIter):
  def one_hot(self, x):
    xs = x.reshape(x.shape[0] * x.shape[1], ) if len(x.shape) == 2 else x
    xs[xs==10429] = 0
    res = np.zeros(list(xs.shape) + [self.n_out],'int32')
    res[np.arange(xs.shape[0]), xs] = 1
    return res.reshape(x.shape[0],x.shape[1],self.n_out) if len(x.shape) == 2 else res

  def __init__(self, filename):  
    self.batches = []  
    h5 = h5py.File(filename, "r")
    lengths = h5["seqLengths"][...].T[0].tolist()
    xin = h5['inputs'][...]
    yin = h5['targets/data']['classes'][...]
    yin[yin==10429] = 0
    self.n_out = h5['targets/size'].attrs['classes']
    self.n_in = xin.shape[1]
    self.n_seqs = len(lengths)
    i = 0
    while i < len(lengths):
      end = min(i+BATCH_SIZE,len(lengths))
      batch_x = np.zeros((MAX_LEN, BATCH_SIZE, xin.shape[1]), 'float32')
      batch_y = np.zeros((MAX_LEN, BATCH_SIZE), 'int8')
      #batch_y = np.zeros((BATCH_SIZE, MAX_LEN), 'int32')
      batch_i = np.zeros((BATCH_SIZE, MAX_LEN), 'int8')
      for j in xrange(end-i):
        batch_x[:lengths[i+j],j] = (xin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        #batch_y[j,:lengths[i+j]] = self.one_hot(yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        batch_y[:lengths[i+j],j] = yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])]
        #batch_y[j * MAX_LEN:j * MAX_LEN + lengths[i+j]] = yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])]
        batch_i[j,:lengths[i+j]] = 1
      self.batches.append((batch_x,batch_y,batch_i,MAX_LEN)) #max(lengths[i:end])))
      i = end
    self.lengths = lengths
    h5.close()
    self.batch_idx = 0

  def next(self):
    if self.batch_idx == len(self.batches):
      self.batch_idx = 0
      return None, None, None, None
    self.batch_idx += 1
    return self.batches[self.batch_idx-1]

  def __iter__(self):
    return self

#train = HDF5DATA('data/train.0001')
train = HDF5DATA('data/train.0002')
valid = HDF5DATA('data/train.0002')

# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 27
display_step = 10

# Network Parameters
n_input = train.n_in # MNIST data input (img shape: 28*28)
n_steps = 250 # timesteps
n_hidden = 512 # hidden layer num of features
n_classes = train.n_out # MNIST total classes (0-9 digits)
num_lstm_layer = 3


def lstm(num_hidden, indata, prev_state, param, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="l%d_i2h" % layeridx)
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="l%d_h2h" % layeridx)
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="l%d_slice" % layeridx)
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def lstm_unroll(num_lstm_layer, 
                num_hidden, dropout=0.,
                concat_decode=True, use_loss=False):
    """unrolled lstm network"""
    with mx.AttrScope(ctx_group='decode'):
        cls_weight = mx.sym.Variable("cls_weight")
        cls_bias = mx.sym.Variable("cls_bias")

    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        with mx.AttrScope(ctx_group='layer%d' % i):
            param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                         i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                         h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                         h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
            state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                              h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)

    # stack LSTM
    hidden = mx.sym.SliceChannel(data=mx.sym.Variable("data"), num_outputs=MAX_LEN, squeeze_axis=0)
    for i in range(num_lstm_layer):
       next_hidden = []
       for t in range(MAX_LEN):
         with mx.AttrScope(ctx_group='layer%d' % i):
           next_state = lstm(n_hidden, indata=hidden[t],
                              prev_state=last_states[i],
                              param=param_cells[i],
                              layeridx=i, dropout=0.)
           next_hidden.append(next_state.h)
           last_states[i] = next_state
       hidden = next_hidden[:]

    sm = []
    labels = mx.sym.SliceChannel(data=mx.sym.Variable("labels"), num_outputs=MAX_LEN, squeeze_axis=0)
    for t in range(MAX_LEN):
      fc = mx.sym.FullyConnected(data=hidden[t],
                               weight=cls_weight,
                               bias=cls_bias,
                               num_hidden=n_classes)
      sm.append(mx.sym.softmax_cross_entropy(fc, labels[t], name="sm"))

    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name="l%d_last_c" % i),
                          h=mx.sym.BlockGrad(state.h, name="l%d_last_h" % i))
        last_states[i] = state

    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = sm + unpack_c + unpack_h
    return mx.sym.Group(list_all)


def is_param_name(name):
    return name.endswith("weight") or name.endswith("bias") or\
        name.endswith("gamma") or name.endswith("beta")


def setup_rnn_model(default_ctx,
                    num_lstm_layer, 
                    num_hidden,
                    batch_size,
                    initializer,
                    group2ctx=None, concat_decode=True,
                    use_loss=False, buckets=None):
    """set up rnn model with lstm cells"""
    max_len = n_steps
    max_rnn_exec = None
    rnn_sym = lstm_unroll(num_lstm_layer=num_lstm_layer,
                          num_hidden=num_hidden)
    arg_names = rnn_sym.list_arguments()
    internals = rnn_sym.get_internals()

    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size,MAX_LEN)
        elif name == "labels":
            input_shapes[name] = (batch_size,MAX_LEN)
        elif name.endswith("label"):
            input_shapes[name] = (batch_size, )
        else:
            pass

    arg_shape, out_shape, aux_shape = rnn_sym.infer_shape(**input_shapes)
    # bind arrays
    arg_arrays = []
    args_grad = {}
    for shape, name in zip(arg_shape, arg_names):
        group = internals[name].attr("ctx_group")
        ctx = group2ctx[group] if group is not None else default_ctx
        arg_arrays.append(mx.nd.zeros(shape, ctx))
        if is_param_name(name):
            args_grad[name] = mx.nd.zeros(shape, ctx)
        if not name.startswith("t"):
            print("%s group=%s, ctx=%s" % (name, group, str(ctx)))
    
    #bind with shared executor
    rnn_exec = None
    rnn_exec = rnn_sym.bind(default_ctx, args=arg_arrays,
                      args_grad=args_grad,
                      grad_req="add", group2ctx=group2ctx)
    max_rnn_exec = rnn_exec
    

    param_blocks = []
    arg_dict = dict(zip(arg_names, rnn_exec.arg_arrays))
    for i, name in enumerate(arg_names):
        if is_param_name(name):
            initializer(name, arg_dict[name])
            param_blocks.append((i, arg_dict[name], args_grad[name], name))
        else:
            assert name not in args_grad

    out_dict = dict(zip(rnn_sym.list_outputs(), rnn_exec.outputs))

    init_states = [LSTMState(c=arg_dict["l%d_init_c" % i],
                         h=arg_dict["l%d_init_h" % i]) for i in range(num_lstm_layer)]

    seq_data = rnn_exec.arg_dict["data"]
    # we don't need to store the last state 
    last_states = None

    seq_outputs = [out_dict["sm_output"]]
    seq_labels = [rnn_exec.arg_dict["labels"]]

    model = LSTMModel(rnn_exec=rnn_exec, symbol=rnn_sym,
                 init_states=init_states, last_states=last_states,
                 seq_data=seq_data, seq_labels=seq_labels, seq_outputs=seq_outputs,
                 param_blocks=param_blocks)
    return model


def train_lstm(model, X_train_batch, X_val_batch,
               update_period, concat_decode, batch_size, use_loss,
               optimizer='adam', half_life=2,max_grad_norm = 5.0, **kwargs):
    opt = mx.optimizer.create(optimizer,
                              **kwargs)

    updater = mx.optimizer.get_updater(opt)
    epoch_counter = 0
    log_period = 28
    last_perp = 10000000.0

    for iteration in range(training_iters):
        nbatch = 0
        train_nll = 0
        tic = time.time()
        while True:
            bx,by,bi,_ = X_train_batch.next()
            if bx is None: break

            m = model
            # reset init state
            for state in m.init_states:
              state.c[:] = 0.0
              state.h[:] = 0.0
              
            head_grad = []
            if use_loss:
              ctx = m.seq_outputs[0].context
              head_grad = [mx.nd.ones((1,), ctx) for x in m.seq_outputs]

            mx.nd.array(bx).copyto(m.seq_data[0])
            m.seq_labels[0][:] = by

            m.rnn_exec.forward(is_train=True)
            seq_loss = [x.copyto(mx.cpu()) for x in m.seq_outputs]
            m.rnn_exec.backward(head_grad)

            # update epoch counter
            epoch_counter += 1
            if epoch_counter % update_period == 0:
                # updare parameters
                norm = 0.
                #for idx, weight, grad, name in m.param_blocks:
                #    grad /= batch_size
                #    l2_norm = mx.nd.norm(grad).asscalar()
                #    norm += l2_norm*l2_norm
                norm = math.sqrt(norm)
                for idx, weight, grad, name in m.param_blocks:
                    #if norm > max_grad_norm:
                    #    grad *= (max_grad_norm / norm)
                    updater(idx, grad, weight)
                    # reset gradient to zero
                    grad[:] = 0.0

            train_nll += sum([x.asscalar() for x in seq_loss])
            nbatch += batch_size
            toc = time.time()
            if epoch_counter % log_period == 0:
                print("Iter [%d] Train: Time: %.3f sec, NLL=%.3f, Perp=%.3f" % (
                    epoch_counter, toc - tic, train_nll / nbatch, np.exp(train_nll / nbatch)))
        # end of training loop
        toc = time.time()
        print("Iter [%d] Train: Time: %.3f sec, NLL=%.3f, Perp=%.3f" % (
            iteration, toc - tic, train_nll / nbatch, np.exp(train_nll / nbatch)))

        val_nll = 0.0
        nbatch = 0
        for bx,by,bi,ml in X_val_batch.batches:
            batch_seq_length = n_steps
            m = model

            # validation set, reset states
            for state in m.init_states:
                state.h[:] = 0.0
                state.c[:] = 0.0

            mx.nd.array(bx).copyto(m.seq_data[0])
            m.seq_labels[0][:] = by

            m.rnn_exec.forward(is_train=False)
            seq_loss = [x.copyto(mx.cpu()) for x in m.seq_outputs]

            
            val_nll += sum([x.asscalar() for x in seq_loss]) / batch_size
            nbatch += batch_size
            
        perp = np.exp(val_nll / nbatch)
        print("Iter [%d] Val: NLL=%.3f, Perp=%.3f" % (
            iteration, val_nll / nbatch, np.exp(val_nll / nbatch)))
        if last_perp - 1.0 < perp:
            opt.lr *= 0.5
            print("Reset learning rate to %g" % opt.lr)
        last_perp = perp
        X_val_batch.reset()
        X_train_batch.reset()


init_c = [('l%d_init_c'%l, (batch_size, n_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, n_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h


#X_train_batch = BucketSentenceIter("./data/ptb.train.txt", dic,
#                                        buckets, batch_size, init_states, model_parallel=True)
#X_val_batch = BucketSentenceIter("./data/ptb.valid.txt", dic,
#                                      buckets, batch_size, init_states, model_parallel=True)


ngpu = 4
# A simple two GPU placement plan
group2ctx = {'embed': mx.gpu(0),
             'decode': mx.gpu(ngpu - 1)}

for i in range(num_lstm_layer):
    group2ctx['layer%d' % i] = mx.gpu(i * ngpu // num_lstm_layer)


model = setup_rnn_model(mx.gpu(), group2ctx=group2ctx,
                             num_lstm_layer=num_lstm_layer,
                             num_hidden=n_hidden,
                             batch_size=batch_size,
                             initializer=mx.initializer.Uniform(0.1))

train_lstm(model, train, valid, 
                half_life=2,
                update_period=1,
                learning_rate=learning_rate,
                batch_size = batch_size,
                wd=0.)
