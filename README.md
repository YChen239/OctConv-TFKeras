# OctConv-TFKeras
Unofficial implementation of Octave Convolutions (OctConv) in TensorFlow / Keras. 

Y. Chen, H. Fang, B. Xu, Z. Yan, Y. Kalantidis, M. Rohrbach, S. Yan, J. Feng. *Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution*. (2019). https://arxiv.org/abs/1904.05049

![](octconv_02.png)

(Update 2019-04-26) Official implementation by MXNet is available : [https://github.com/facebookresearch/OctConv](https://github.com/facebookresearch/OctConv) 

# Usage
```python
from oct_conv2d import OctConv2D
# high, low = some tensors or inputs
high, low = OctConv2D(filters=ch, alpha=alpha)([high, low])
```

# Colab Notebook
**Train OctConv ResNet** (TPU)  
https://colab.research.google.com/drive/1MXN46mhCk6s-G_nfJrH1B6_8GXh-a_QH

**Measuring prediction time** (CPU)  
https://colab.research.google.com/drive/12MdVXyB9K3FnpzYNmyc3qu5s59-53WNE

# CIFAR-10
Experimented with Wide ResNet (N = 4, k = 10). Train with colab TPUs.


## Reference Code
- Densenet(https://github.com/cmasch/densenet)
- Oct-Resnent(https://github.com/koshian2/OctConv-TFKeras)
