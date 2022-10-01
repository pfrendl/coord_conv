# coord_conv
An evaluation of the approach from the CoordConv Uber [paper](https://arxiv.org/abs/1807.03247).
Their GitHub page: [https://github.com/uber-research/CoordConv/](https://github.com/uber-research/CoordConv/)

After skimming through the implementation, I found that the coordinate regression task had been implemented with a model that has no downsampling.
This is not evident from the [blog post](https://www.uber.com/en-HU/blog/coordconv/) and it surprised me, since the approach is not very practical.

This repository implements a version of the pixel coordinate regression tasks that has downsampling in it. In addtition to the convolutional networks, I have also added a model similar to the one in the Uber implementation, but this one has 1x1 sized kernels, since 3x3 kernels seem to provide no further benefit. I refer to it as an attention-based model, since it can be interpreted as a query mechanism with a single query over pixel-wise feature vectors containing color and position information.

By running main.py, I get the following outputs:
![overfit_grid](https://user-images.githubusercontent.com/6968154/193423950-ea2282fd-8cec-4566-8c88-120864d98f06.png)
![uniform_split](https://user-images.githubusercontent.com/6968154/193423953-a21d8cd7-c36f-4d94-bdcb-62210a58b7c0.png)
![quadrant_split](https://user-images.githubusercontent.com/6968154/193423955-e1193349-09b4-40d1-a019-aeef26c064b0.png)

CoordConv seems to be disadvantageous when downsampling is involved, and none of the models can solve the quadrant split experiment. Moreover, regular convolution perfroms quite well in the overfitting task and the uniform split task. The attention-based model exhibits a partial generalization ability on the quadrant split task.
