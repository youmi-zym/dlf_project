# Project for [PhD Course: A Comparative Introduction to Deep Learning Frameworks: TensorFlow, PyTorch and JAX 2021-22](https://www.mircomusolesi.org/courses/DL-PhD21-22/DL-PhD21-22-main/)

## A Stereo Matching Network Implemented in TensorFlow and PyTorch

### Running !!!

[run.ipynb](https://github.com/youmi-zym/dlf_project/blob/main/run.ipynb)

### Method Description

In this project, we implemented a stereo matching network that exploits stack hourglass network for cost aggregation. This architecture achieves high performance on depth prediction on KITTI benchmark.

### Dataset

We utilize the training split of [KITTI-2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) for training and evaluation. 

#### Pre-train
The stereo matching network is pretrained on the frist 160 images of KITTI-2015 for 300 epochs, the pre-trained checkpoint can be found at [ckpt](https://github.com/youmi-zym/dlf_project/tree/main/ckpt) sub-folder.

Specifically, the script for pre-training in PyTorch could be run with following command:

```bash
python torch_train.py
```
Besides, the TensorFlow version could be run with:

```bash
CUDA_VISIBLE_DEVICES=0 python tf_train.py
```

## Contact

If you have any issue or question, email to [Youmin Zhang](https://youmi-zym.github.io/): youmin.zhang2@unibo.it 
