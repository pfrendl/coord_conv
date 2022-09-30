# coord_conv
An evaluation of the approach from the CoordConv Uber [paper](https://arxiv.org/abs/1807.03247).
Their GitHub page: [https://github.com/uber-research/CoordConv/](https://github.com/uber-research/CoordConv/)

After skimming through the implementation, I found that the coordinate regression task had been implemented with a model that has no downsampling.
This is not evident from the [blog post](https://www.uber.com/en-HU/blog/coordconv/) and it surprised me, since the approach is not very practical.

This repository implements a version of the pixel coordinate regression tasks that has downsampling in it.

By running main.py, I get the following outputs:
![overfit_grid](https://user-images.githubusercontent.com/6968154/193352301-50613ffc-1539-4b3e-ba93-0162799f5d7a.png)
![uniform_split](https://user-images.githubusercontent.com/6968154/193352318-ff2e756c-488d-4e54-bcdf-cab14e98c835.png)
![quadrant_split](https://user-images.githubusercontent.com/6968154/193352339-5e35ee39-65fd-46e7-b50f-d19e304322a4.png)

CoordConv seems to be disadvantageous when downsampling is involved, and none of the models can solve the quadrant split experiment. Moreover, regular convolution perfroms much better than indicated by Uber.
