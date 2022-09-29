# coord_conv
An evaluation of the approach from the CoordConv Uber [paper](https://arxiv.org/abs/1807.03247).
Their GitHub page: [https://github.com/uber-research/CoordConv/](https://github.com/uber-research/CoordConv/)

After skimming through the implementation, I found that the coordinate regression task had been implemented with a model that has no downsampling.
This is not evident from the [blog post](https://www.uber.com/en-HU/blog/coordconv/) and it surprised me, since the approach is not very practical.

This repository implements a version of the pixel coordinate regression task that has downsampling in it.

By running main.py, I get the following output:
![test](https://user-images.githubusercontent.com/6968154/193093468-007b6983-0ec8-42a2-9c31-701fc7d475a9.png)

CoordConv seems to be disadvantageous when downsampling is involved.
