# GLUCLSON
`GluClsOn` is a toy project to share:
1. Some of classification experiment results with Gluon models. 
2. My re-implementation of some classification networks with Gluon.
3. Tools of ImageNet classification with Gluon.


### Gluon Pretrained Model Test ([exp](./exp/model_zoo_test.ipynb)|[result](.exp/model_zoo_test_result.csv))

Part of performance test results of [Gluon pretrained models](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/model_zoo/model_store.py) on **ImageNet** validation dataset:

|    Model    | Top-1 err.(%) | Top-5 err.(%) | 
|-------------|---------------|---------------|
| resnet152_v2 | 21.98 | 6.08 |
| densenet161 | 22.39 | 6.07 |
| resnet101_v2 | 22.75 | 6.35 |
| resnet152_v1 | 23.04 | 6.48 |
| densenet201 | 23.10 | 6.53 |
| resnet101_v1 | 23.37 | 6.66 |
| densenet169 | 23.84 | 6.82 |
| resnet50_v2 | 24.04 | 7.19 |
| resnet50_v1 | 24.92 | 7.48 |
| densenet121 | 25.44 | 7.80 |
| resnet34_v2 | 27.44 | 8.90 |
| vgg19_bn | 28.92 | 9.66 |
| resnet34_v1 | 29.46 | 9.81 |
| vgg16_bn | 29.71 | 9.93 |
| mobilenet1.0 | 29.78 | 10.36 |
