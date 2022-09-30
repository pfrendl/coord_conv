# coord_conv
An evaluation of the approach from the CoordConv Uber [paper](https://arxiv.org/abs/1807.03247).
Their GitHub page: [https://github.com/uber-research/CoordConv/](https://github.com/uber-research/CoordConv/)

After skimming through the implementation, I found that the coordinate regression task had been implemented with a model that has no downsampling.
This is not evident from the [blog post](https://www.uber.com/en-HU/blog/coordconv/) and it surprised me, since the approach is not very practical.

This repository implements a version of the pixel coordinate regression tasks that has downsampling in it.

By running main.py, I get the following outputs:
![overfit_grid](https://user-images.githubusercontent.com/6968154/193356141-59ba6ba6-2eb9-404c-9470-7603c794cc0f.png)
![uniform_split](https://user-images.githubusercontent.com/6968154/193356154-11ebbf4e-dba0-4f5c-96ab-b877d31aea3a.png)
![quadrant_split](https://user-images.githubusercontent.com/6968154/193356165-624545b7-d590-4b45-b5bd-04a199a1a14e.png)

CoordConv seems to be disadvantageous when downsampling is involved, and none of the models can solve the quadrant split experiment. Moreover, regular convolution perfroms much better than indicated by Uber.
