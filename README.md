# Behavioral-Cloning

This repo contains code for a project I did as a part of [Udacity's Self Driving Car Nano Degree Program](https://www.udacity.com/drive). We had to train a car to drive itself in a simulator. The car was trained to drive itself using a deep neural network.

# Datset

I used the dataset provided by Udacity which are about 8000 images. More images can be generated using Udacity's simulator.

The dataset contains JPG images of dimensions 160x320x3. Here are some sample images from the dataset.

![Sample Images](resource _for_readme/cameraimages.JPG)

# Steering Angle Histogram

By observing this histogram we can say that much of the dataset has 0 steering angle and due to this model may be biased towards it.

![Steering Angle Histogram](resource _for_readme/histogram.JPG)

# Augmentation Techniques Used and Architecture

![Nvidia's Architecture](resource _for_readme/nvidia_architecture.png)

I have to thank this [NVIDIA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and [this blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28) for suggesting these techniques.

1.Flip the images horizontally.
2.Brightness Adjustment.
3.Adding random shadows.

Below is an example of generated data from the image source.

![Sample Generated Data](resource _for_readme/preprocessing.JPG)

#Architecture Specification:

- Input Layer is 66 x 220 x 3
- Normalization Layer
- Convolutional Layer 1: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 3 x 66 x 200
- Convolutional Layer 2: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 24 x 31 x 98
- Convolutional Layer 3: 5 x 5 Kernel, 2 x 2 stride, valid padding, Output: 36 x 14 x 47
- Convolutional Layer 4: 5 x 5 Kernel, 3 x 3 stride, valid padding, Output: 48 x 5 x 22
- Convolutional Layer 5: 5 x 5 Kernel, 3 x 3 stride, valid padding, Output: 64 x 3 x 20
- Flatten Layer 1: Output: 64 x 1 x 18
- Fully-Connected Layer 1: Output 1152 neurons
- Fully-Connected Layer 2: Output 100 neurons
- Fully-Connected Layer 3: Output 50 neurons
- Fully-Connected Layer 3: Output 10 neurons
- Final Output: 1

Here normalization is used to avoid over fitting.

# Preproceesing Images
1. I noticed that the bonet of the car is visible in the lower portion of the image.
2. And also the portion above the horizon can also be ignored.
We can crop the image and remove this.

![Final Image](resource _for_readme/crop_image.JPG)

There are many hyper-parameters to tune:

- number of iterations
- number of epochs
- batch size
- Adam optimizer parameters (learning rate, beta\_1, beta\_2, and decay)

I observed that increasing the epochs does not necessarily reduce loss.So by trial and error I tuned the parameters.

# Simulation on the Test Track

Here is the [Driving Simulation on the Test Track](https://youtu.be/ZrM7kzP8yzM).
