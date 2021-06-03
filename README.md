# Equi-RC
Equivariant layers for RC-complement symmetry in DNA sequence data

This is a repository that implements the layers as described in "Reverse-Complement Equivariant Networks for DNA Sequences" in Keras and Pytorch.
The simplest way to use it is to include the appropriate standalone python script in your code.

## Setup and notes
Just install Keras or Pytorch and you can start importing the layers.


The reg_in, reg_out arguments should be understood as the number of cycles.
This correspond to half the input dimension (so for the input, we have 4 nucleotides, so reg=2)

The a_n are of the dimensions of type +1, the b_n of type -1

## Examples
### Keras
This class used for the Binary Prediction task is implemented as an example.
One can refer to this implementation and for testing, simply run :
```
python keras_example.py
```

### Pytorch
The equivalent class is also written in Pytorch, and can be ran with :
```
python pytorch_example.py
```
