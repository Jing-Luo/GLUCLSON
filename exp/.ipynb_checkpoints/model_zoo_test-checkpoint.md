# Validation of Gluon Models on ImageNet

We are about to test the performance of gluon pretrained models in the [model zoo](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon/model_zoo/vision) on ImageNet validation dataset. The validation dataset consists of 50,000 images, 50 per class.

To begin with, import the necessary modules.

```{.python .input  n=41}
import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data import vision
from mxnet.gluon.data.vision import transforms

import shutil
import time
import os
import pandas as pd
```

```{.json .output n=41}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/usr/lib/python3/dist-packages/pandas/__init__.py:7: DeprecationWarning: bad escape \\s\n  from pandas import hashtable, tslib, lib\n"
 }
]
```

## Data Download

First we need to prepare the [validation images](http://www.image-net.org/download-images) and preprocess them. Due to the black magic of Gluon, we do not need to transform the JPEG files to `rec` format.

The DataLoader in Gluon requires the images arranged in the subdir named after the image label. However, the validation images are stored in one folder, so we need to create a subdir for every label, and put a symbol link to the image in the every subdir.

```{.python .input  n=2}
val_path = 'val'
image_path = '/unsullied/sharefs/luojing/exp/Dataset/ImageNet2012Val/ILSVRC2012_img_val' #use absolute address
synsets_file = open('../basics/synsets.txt', 'r')
val_file = open('../basics/val.txt', 'r')

if os.path.exists(val_path):
    shutil.rmtree(val_path)

synsets = [line.rstrip('\n') for line in synsets_file.readlines()]

for line in val_file.readlines():
    fname, idx = line.split()
    label_path = '%s/%s' % (val_path, synsets[int(idx)])
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    os.symlink('%s/%s' % (image_path, fname), '%s/%s' % (label_path, fname))
```

The common benchmark of ImageNet validation is Top 1/5 error of `224x` on `256x` Crop. That is how we define the tranform function.

```{.python .input  n=3}
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

We use `ImageFolderDataset` to read the validation dataset.

```{.python .input  n=40}
batch_size = 25
num_gpus = 8
batch_size *= max(1, num_gpus)
val_total = 50000
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = 10

val_dataset = vision.ImageFolderDataset(val_path)
val_data = gluon.data.DataLoader(
        val_dataset.transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

```{.json .output n=40}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Traceback (most recent call last):\n  File \"/usr/lib/python3.5/multiprocessing/queues.py\", line 247, in _feed\n    send_bytes(obj)\n  File \"/usr/lib/python3.5/multiprocessing/connection.py\", line 200, in send_bytes\n    self._send_bytes(m[offset:offset + size])\n  File \"/usr/lib/python3.5/multiprocessing/connection.py\", line 404, in _send_bytes\n    self._send(header + buf)\n  File \"/usr/lib/python3.5/multiprocessing/connection.py\", line 368, in _send\n    n = write(self._handle, buf)\nBrokenPipeError: [Errno 32] Broken pipe\n"
 }
]
```

## Load Models and Test

```{.python .input  n=42}
from mxnet.gluon.model_zoo.model_store import _model_sha1

test_result = []
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

for model in sorted(_model_sha1.keys()):
    if model == 'inceptionv3':
        continue
    net = models.get_model(model, pretrained=True, ctx=ctx)
    acc_top1.reset()
    acc_top5.reset()    
    for _, batch in enumerate(val_data):       
        data = gluon.utils.split_and_load(batch[0], ctx)
        label = gluon.utils.split_and_load(batch[1], ctx)       
        outputs = [net(X) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        # print_str = 'Top 1 Err: %4f \t Top 5 Err: %4f '%(1 - top1, 1 - top5)
        # pbar.set_description("%s" % print_str)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    print('Model: %s \t Top 1 Err: %4f \t Top 5 Err: %4f '%(model, 1 - top1, 1 - top5))
    test_result.append((model, 1 - top1, 1 - top5))   
        
```

```{.json .output n=42}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Model: alexnet \t Top 1 Err: 0.470920 \t Top 5 Err: 0.233480 \nModel: densenet121 \t Top 1 Err: 0.254400 \t Top 5 Err: 0.078040 \nModel: densenet161 \t Top 1 Err: 0.223880 \t Top 5 Err: 0.060680 \nModel: densenet169 \t Top 1 Err: 0.238380 \t Top 5 Err: 0.068240 \nModel: densenet201 \t Top 1 Err: 0.231000 \t Top 5 Err: 0.065260 \nModel: mobilenet0.25 \t Top 1 Err: 0.499980 \t Top 5 Err: 0.255280 \nModel: mobilenet0.5 \t Top 1 Err: 0.380900 \t Top 5 Err: 0.159200 \nModel: mobilenet0.75 \t Top 1 Err: 0.334320 \t Top 5 Err: 0.127500 \nModel: mobilenet1.0 \t Top 1 Err: 0.297760 \t Top 5 Err: 0.103600 \nModel: resnet101_v1 \t Top 1 Err: 0.233680 \t Top 5 Err: 0.066580 \nModel: resnet101_v2 \t Top 1 Err: 0.227520 \t Top 5 Err: 0.063480 \nModel: resnet152_v1 \t Top 1 Err: 0.230380 \t Top 5 Err: 0.064820 \nModel: resnet152_v2 \t Top 1 Err: 0.219760 \t Top 5 Err: 0.060840 \nModel: resnet18_v1 \t Top 1 Err: 0.336340 \t Top 5 Err: 0.126860 \nModel: resnet18_v2 \t Top 1 Err: 0.310500 \t Top 5 Err: 0.113540 \nModel: resnet34_v1 \t Top 1 Err: 0.294600 \t Top 5 Err: 0.098060 \nModel: resnet34_v2 \t Top 1 Err: 0.274400 \t Top 5 Err: 0.089000 \nModel: resnet50_v1 \t Top 1 Err: 0.249180 \t Top 5 Err: 0.074800 \nModel: resnet50_v2 \t Top 1 Err: 0.240360 \t Top 5 Err: 0.071920 \nModel: squeezenet1.0 \t Top 1 Err: 0.459460 \t Top 5 Err: 0.226700 \nModel: squeezenet1.1 \t Top 1 Err: 0.477840 \t Top 5 Err: 0.237060 \nModel: vgg11 \t Top 1 Err: 0.348600 \t Top 5 Err: 0.135040 \nModel: vgg11_bn \t Top 1 Err: 0.327040 \t Top 5 Err: 0.121760 \nModel: vgg13 \t Top 1 Err: 0.337020 \t Top 5 Err: 0.127020 \nModel: vgg13_bn \t Top 1 Err: 0.328260 \t Top 5 Err: 0.120140 \nModel: vgg16 \t Top 1 Err: 0.315920 \t Top 5 Err: 0.113100 \nModel: vgg16_bn \t Top 1 Err: 0.297100 \t Top 5 Err: 0.099320 \nModel: vgg19 \t Top 1 Err: 0.305620 \t Top 5 Err: 0.107440 \nModel: vgg19_bn \t Top 1 Err: 0.289160 \t Top 5 Err: 0.096620 \n"
 }
]
```

```{.python .input  n=43}
summary = pd.DataFrame(test_result, columns=['model', 'top_1_err', 'top_5_err'])
summary = summary.sort_values('top_1_err')
summary.head()
```

```{.json .output n=43}
[
 {
  "data": {
   "text/html": "<div>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>model</th>\n      <th>top_1_err</th>\n      <th>top_5_err</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>12</th>\n      <td>resnet152_v2</td>\n      <td>0.21976</td>\n      <td>0.06084</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>densenet161</td>\n      <td>0.22388</td>\n      <td>0.06068</td>\n    </tr>\n    <tr>\n      <th>10</th>\n      <td>resnet101_v2</td>\n      <td>0.22752</td>\n      <td>0.06348</td>\n    </tr>\n    <tr>\n      <th>11</th>\n      <td>resnet152_v1</td>\n      <td>0.23038</td>\n      <td>0.06482</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>densenet201</td>\n      <td>0.23100</td>\n      <td>0.06526</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "           model  top_1_err  top_5_err\n12  resnet152_v2    0.21976    0.06084\n2    densenet161    0.22388    0.06068\n10  resnet101_v2    0.22752    0.06348\n11  resnet152_v1    0.23038    0.06482\n4    densenet201    0.23100    0.06526"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=46}
for i, (model, top1, top5) in summary.iterrows():
    print('| %s | %.2f | %.2f |'%(model, top1 * 100, top5 * 100))
```

```{.json .output n=46}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "| resnet152_v2 | 21.98 | 6.08 |\n| densenet161 | 22.39 | 6.07 |\n| resnet101_v2 | 22.75 | 6.35 |\n| resnet152_v1 | 23.04 | 6.48 |\n| densenet201 | 23.10 | 6.53 |\n| resnet101_v1 | 23.37 | 6.66 |\n| densenet169 | 23.84 | 6.82 |\n| resnet50_v2 | 24.04 | 7.19 |\n| resnet50_v1 | 24.92 | 7.48 |\n| densenet121 | 25.44 | 7.80 |\n| resnet34_v2 | 27.44 | 8.90 |\n| vgg19_bn | 28.92 | 9.66 |\n| resnet34_v1 | 29.46 | 9.81 |\n| vgg16_bn | 29.71 | 9.93 |\n| mobilenet1.0 | 29.78 | 10.36 |\n| vgg19 | 30.56 | 10.74 |\n| resnet18_v2 | 31.05 | 11.35 |\n| vgg16 | 31.59 | 11.31 |\n| vgg11_bn | 32.70 | 12.18 |\n| vgg13_bn | 32.83 | 12.01 |\n| mobilenet0.75 | 33.43 | 12.75 |\n| resnet18_v1 | 33.63 | 12.69 |\n| vgg13 | 33.70 | 12.70 |\n| vgg11 | 34.86 | 13.50 |\n| mobilenet0.5 | 38.09 | 15.92 |\n| squeezenet1.0 | 45.95 | 22.67 |\n| alexnet | 47.09 | 23.35 |\n| squeezenet1.1 | 47.78 | 23.71 |\n| mobilenet0.25 | 50.00 | 25.53 |\n"
 }
]
```

```{.python .input  n=47}
summary.to_csv('model_zoo_test_result.csv', index=None)
```
