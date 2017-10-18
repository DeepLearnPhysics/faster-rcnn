# faster-rcnn
A Tensorflow implementation of faster RCNN detection framework toward 3D object detection. This repository is based on the original python Caffe implementation of faster RCNN by Ross Girshick available [here](https://github.com/rbgirshick/py-faster-rcnn) and implementation to Tensorflow by Xinlei Chen available [here](https://github.com/endernewton/tf-faster-rcnn). While much of the code follows Xinlei's implementation, the reason why we did not fork is because we plan to extend further to 3D. This involves much reorganization of code/directory structures and departs from the goal of replicating py-caffe implementation. Note changes made by Xinlei (summarized in their paper [An Implementation of Faster RCNN with Study for Region Sampling](https://arxiv.org/pdf/1702.02138.pdf)) is inherited as is. 

**Note**:
This implementation followed Xinlei's method step by step, and may still lack some functionalities and/or performance.
The current goal is to reproduce the results in Xinlei's paper. The network does get trained with vgg implementation (resnet not yet tried).

### Prerequisites
  - Tensorflow: currently tested with v1.3. If concerned, Xinlei's Tensorflow [fork](https://github.com/endernewton/tensorflow) implements the original ROI pooling in py-caffe.
  - Python packages: `cython`, `opencv-python`, `easydict`

### Installation
1. Clone the repository
  ```Shell
  git clone https://github.com/DeepLearnPhysics/faster-rcnn.git
  ```
2. Source configure script (you have to do this each time you login to a new shell).
  ```Shell
  cd faster-rcnn
  source configure.sh
  ```

3. Update your -arch in setup script to match your GPU
  ```Shell
  # Change the GPU architecture (-arch) if necessary
  vim setup.py
  ```

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

  **Note**: You are welcome to contribute the settings on your end if you have made the code work properly on other GPUs.

4. Build the Cython modules
  ```Shell
  make clean
  make
  ```

### Setup data for training (py-faster-rcnn has more detailed instructions [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models))
1. Download the training, validation, test data and VOCdevkit

   ```Shell
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```Shell
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```Shell
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4. Create symlinks for the PASCAL VOC dataset

   ```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] for coco data set see py-faster-rcnn instruction [here](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data)

### Train your own model
1. Download pre-trained models and weights. The current code support VGG16. Pre-trained models are provided by slim, you can get the pre-trained models [here](https://github.com/tensorflow/models/tree/master/slim#pre-trained-models) and set them in the ``data`` folder. For example for VGG16 model, you can set up like:
   ```Shell
   cd data
   wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
   tar -xzvf vgg_16_2016_08_28.tar.gz
   mv vgg_16.ckpt vgg16.ckpt
   cd ..
   ```
2. Train (and test, evaluation)
  ```Shell
  python example/train_coco.py
  ```

3. Visualization with Tensorboard
  ```Shell
  tensorboard --logdir=tensorboard/log &
  ```

By default, trained networks are saved under:

```
output/[NET]
```

Tensorboard information for train and validation is saved under:

```
tensorboard/[NET]/
```

### Citation
For citation, please consider citing Xinlei's work

    @article{chen17implementation,
        Author = {Xinlei Chen and Abhinav Gupta},
        Title = {An Implementation of Faster RCNN with Study for Region Sampling},
        Journal = {arXiv preprint arXiv:1702.02138},
        Year = {2017}
    }

as well as faster RCNN citation:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

