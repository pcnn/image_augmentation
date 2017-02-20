# Image augmentation package
Augmenting datasets of images is a common practice in training convolutional neural networks. Several methods for augmenting and degrading an image are implemented in this repository. This package can be used for modifying miages on-the-fly or modifying and storing them on disks.

The image in this repository is taken from [http://www.drodd.com/](http://www.drodd.com/images15/nature13.jpg)
![alt tag](nature.jpg)

## Requirements
This package depends on OpenCV and Numpy libraries.

## Examples
Following are result of degrading the above images using functions in *image_degrade* module.
```python
gaussian_noise(im,mu=0, sigma=30, sparse_prob=0.1)
```
![alt tag](degraded_gaussian.png)