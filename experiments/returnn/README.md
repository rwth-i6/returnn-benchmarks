Edit your `~/.theanorc` to contain at least this:

    [global]
    device = cpu
    floatX = float32

For my system, I also added this:

    # http://deeplearning.net/software/theano/install_ubuntu.html
    # http://stackoverflow.com/questions/11987325/theano-fails-due-to-numpy-fortran-mixup-under-ubuntu
    [blas]
    ldflags = -lblas -lgfortran

Then run:

    ./run.sh <config>

You might need to fix the file `theano-cuda-activate.sh` for your system.
If you want to run it on CPU, you can source `theano-cpu-activate.sh` instead.

