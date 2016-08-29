
# Needed for Theano.
source /u/zeyer/py-envs/py2-theano/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/atlas-base/

export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH  # also needed for sprint

export THEANO_FLAGS="device=cpu"
export THEANO_FLAGS="$THEANO_FLAGS,base_compiledir=/var/tmp/$LOGNAME/theano/,compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s--cpu"
