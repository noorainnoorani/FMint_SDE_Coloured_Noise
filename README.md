## installation

First create a conda environment 

```
conda create -n icon python=3.10 -y
conda activate icon
```


To make sure that tensorflow detect gpus, first install the following 

```
conda install -c conda-forge cudatoolkit=11.8
pip install tensorflow==2.15.0
conda install -c conda-forge cudnn=8.9.7.29
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

`cudnn=8.9.2.26` is not available from the current conda-forge or NVIDIA
linux-64 channels. The `8.9.7.29` build is available on conda-forge and keeps
the install on cuDNN 8.9 for the JAX `cuda12.cudnn89` wheel below.

Then install necessary packages
```
pip install -r env.txt
pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Run the following to check 
```
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
```

Each time after activating the environment, make sure that tensorflow is able to detect all eight GPUs. If not try running
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')))"
```
