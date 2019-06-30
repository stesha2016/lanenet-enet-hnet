# LaneNet-HNet-Detection
Use tensorflow to implement a Deep Neural Network for real time lane detection mainly based on the IEEE IV conference 
paper "Towards End-to-End Lane Detection: an Instance Segmentation Approach".You can refer to their paper for details 
https://arxiv.org/abs/1802.05591. This model consists of a encoder-decoder stage, binary semantic segmentation stage 
and instance semantic segmentation using discriminative loss function for real time lane detection task.

The main network architecture is as follows:

`Network Architecture`
![NetWork_Architecture](/data/source_image/network_architecture.png)

## Installation
This software has only been tested on ubuntu 16.04(x64), python3.5, cuda-9.0, cudnn-7.0 with a GTX-1070 GPU. 
To install this software you need tensorflow 1.10.0 and other version of tensorflow has not been tested but I think 
it will be able to work properly in tensorflow above version 1.10. Other required package you may install them by

```
pip3 install -r requirements.txt
```

## Test model
In this repo I uploaded a model trained on tusimple lane dataset [Tusimple_Lane_Detection](http://benchmark.tusimple.ai/#/).
The deep neural network inference part can achieve around a 50fps which is similar to the description in the paper. But
the input pipeline I implemented now need to be improved to achieve a real time lane detection system.

The trained lanenet model weights files are stored in 
[new_lanenet_model_file](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0). You can 
download the model and put them in folder model/tusimple_lanenet/

Added hnet , so you can test a single image on the trained model as follows

```
python lanenet_hnet_predict.py --image_path ./data/tusimple_data/training/gt_image/0001.png --lanenet_weights ./model/tusimple_lanenet/tusimple_lanenet_enet_2019-05-08-11-11-01.ckpt-100000 --hnet_weights ./model/hnet/hnet-2000
```
The results are as follows:

* The first example

`Test Input Image`

![Test Input](/data/source_image/one/origin_image.png)

`Test Lane Binary Segmentation Image`

![Test Lane_Binary_Seg](/data/source_image/one/predict_binary.png)

`Test Lane Instance Segmentation Image`

![Test Lane_Instance_Seg](/data/source_image/one/predict_instance.png)

`Test Lane Clustered Image`

![Test Lane_Clustered](/data/source_image/one/predict_lanenet.png)

`Test Hnet Image`

![Test Hnet](/data/source_image/one/predict_hnet.png)

`Test Warped Image`

![Test Warped](/data/source_image/one/warped.png)

* The second example

`Test Input Image`

![Test Input](/data/source_image/two/origin_image.png)

`Test Lane Binary Segmentation Image`

![Test Lane_Binary_Seg](/data/source_image/two/predict_binary.png)

`Test Lane Instance Segmentation Image`

![Test Lane_Instance_Seg](/data/source_image/two/predict_instance.png)

`Test Lane Clustered Image`

![Test Lane_Clustered](/data/source_image/two/predict_lanenet.png)

`Test Hnet Image`

![Test Hnet](/data/source_image/two/predict_hnet.png)

`Test Warped Image`

![Test Warped](/data/source_image/two/warped.png)

If you want to test the model on a whole dataset you may call
```
python tools/test_lanenet.py --is_batch True --batch_size 2 --save_dir data/tusimple_test_image/ret 
--weights_path path/to/your/model_weights_file 
--image_path data/tusimple_test_image/
```
If you set the save_dir argument the result will be saved in that folder or the result will not be saved but be 
displayed during the inference process holding on 3 seconds per image. I test the model on the whole tusimple lane 
detection dataset and make it a video. You may catch a glimpse of it bellow.
`Tusimple test dataset gif`
![tusimple_batch_test_gif](/data/source_image/lanenet_batch_test.gif)

## Train your own model
#### Data Preparation
Firstly you need to organize your training data refer to the data/training_data_example folder structure. And you need 
to generate a train.txt and a val.txt to record the data used for training the model. 

The training samples are consist of three components. A binary segmentation label file and a instance segmentation label
file and the original image. The binary segmentation use 255 to represent the lane field and 0 for the rest. The 
instance use different pixel value to represent different lane field and 0 for the rest.

All your training image will be scaled into the same scale according to the config file.
```angular2html
python tools/generate_tusimple_dataset.py --src_dir path/to/your/unzipped/file
```

#### Train model
#### LaneNet
In my experiment the training epochs are 200000, batch size is 8, initialized learning rate is 0.0005 and decrease by 
multiply 0.1 every 100000 epochs. About training parameters you can check the config/global_config.py for details. 
You can switch --net argument to change the base encoder stage. If you choose --net vgg then the vgg16 will be used as 
the base encoder stage and a pretrained parameters will be loaded and if you choose --net dense then the dense net will 
be used as the base encoder stage instead and no pretrained parameters will be loaded. And you can modified the training 
script to load your own pretrained parameters or you can implement your own base encoder stage. 
 * [Modify] Add UNET backbone as mentioned in paper. UNet is a real time architecture network.
 * [Modify] Add gradients clip in case instance loss explored to nan.
You may call the following script to train your own model

```
python train_lanenet.py --net enet --dataset_dir ./data/tusimple_data/training/
```
You can also continue the training process from the ckpt by
```
python train_lanenet.py --net enet --dataset_dir ./data/tusimple_data/training/ --weights_path ./model/lanenet/tusimple_lanenet_enet_2019-04-16-13-36-00.ckpt-2000
```

You may monitor the training process using tensorboard tools

During my experiment the `Total loss` drops as follows:  
![Training loss](/data/source_image/total_loss.png)

The `Binary Segmentation loss` drops as follows:  
![Training binary_seg_loss](/data/source_image/binary_seg_loss.png)

The `Instance Segmentation loss` drops as follows:  
![Training instance_seg_loss](/data/source_image/instance_seg_loss.png)

#### HNet
Training HNet from scratch we have two steps:
 * Training Hnet pre-loss for H matrix to an initialized value.
 ```angular2html
 python hnet_train.py --phase pretrain

```
 ```angular2html
 python hnet_train.py --phase pretrain --pre_hnet_weights ./model/hnet/pre_hnet-9999

```
 * Training Hnet loss to fine tuning H matrix from an initialized value.
  ```angular2html
 python hnet_train.py --phase train --pre_hnet_weights ./model/hnet/pre_hnet-19999

```

## Experiment
The accuracy during training process rises as follows: 
![Training accuracy](/data/source_image/accuracy.png)

Please cite my repo [lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection) if you use it.

## Recently updates 2018.11.10
Adjust some basic cnn op according to the new tensorflow api. Use the 
traditional SGD optimizer to optimize the whole model instead of the
origin Adam optimizer used in the origin paper. I have found that the
SGD optimizer will lead to more stable training process and will not 
easily stuck into nan loss which may often happen when using the origin
code.

I have uploaded a new lanenet model trained on tusimple dataset using the
new code here [new_lanenet_model_file](https://www.dropbox.com/sh/tnsf0lw6psszvy4/AAA81r53jpUI3wLsRW6TiPCya?dl=0).
You may download the new model weights and update the new code. To update
the new code you just need to

```
git pull origin master
```
The rest are just the same as which mentioned above. And recently I will 
release a new model trained on culane dataset.

## Command
 * test
 ```
 python test_lanenet.py --image_path ./data/tusimple_data/training/gt_image/0001.png --weights_path ./model/tusimple_lanenet/tusimple_lanenet_enet_2019-04-16-14-10-07.ckpt-0
 ```

## TODO
- [x] Add Enet backbone for encoder and decoder.
- [x] Add Enet binary and instance loss.
- [x] Training the model on different dataset
- ~~[ ] Adjust the lanenet hnet model and merge the hnet model to the main lanenet model~~
- [ ] Change the normalization function from BN to GN
