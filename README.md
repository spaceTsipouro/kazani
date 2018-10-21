# Chaotic measurement detection & Fractal reshape
A reasonable starting point for the title:
-  In this repo we detect the self similarity dimension of an RGB image using box counting
-  We train an Variational Autoencoder to fractal Images for having fractal features
-  We wanted to use these features in order the way deep dream does in order to enchance the "fractality" feature of any image.

Requirements:
Scipy, numpy, tensorflow

## Usage:

To measure how chaotic an image is:
```
python2.7 fractality_detector.py path_to_jpg_image
```

To train the autoencoder:
```
python2.7 fractal_autoencoder.py
```