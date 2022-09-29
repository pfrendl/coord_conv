# coord_conv
An evaluation of the approach from the CoordConv Uber [paper](https://arxiv.org/abs/1807.03247).

Uber paper GitHub page: [https://github.com/uber-research/CoordConv/](https://github.com/uber-research/CoordConv/)

After skimming through the implementation, I found that the coordinate regression task had been implemented with a model that has no downsampling.
This was not evident from the [blog post](https://www.uber.com/en-HU/blog/coordconv/), and this surprised me, since this solution is not very practical.

This repository implements a version of the pixel coordinate regression task that has downsampling in it.

By running main.py, I get the following output:
![test](https://user-images.githubusercontent.com/6968154/192915700-45f7cf9d-e823-4819-888b-d9d158e3c31f.png)

CoordConv seems to be disadvantageous when downsampling is involved.
