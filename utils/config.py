import mxnet as mx
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.block import HybridBlock
import os


class Normal_Y(HybridBlock):

    def __init__(self):
        super(Normal_Y, self).__init__()

    def hybrid_forward(self, F, x):
        max_v = F.max(x.transpose(()).reshape((0, -1)), axis=1, keepdims=True)
        min_v = F.min(x.reshape((0, -1)), axis=1, keepdims=True)
        scale = max_v - min_v
        return (x - min_v) / scale * 255


class Config(object):
    classes = 3
    epochs = 9
    lr = 0.001
    # orig image size: 1936 x 1216
    scale = 1.592
    im_size = 224

    momentum = 0.9
    wd = 0.0001

    lr_factor = 0.1
    lr_steps = [6, 8, 9, np.inf]

    num_gpus = 4
    num_workers = 4
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    jitter_param = 0.4
    lighting_param = 0.1

    # model_name = 'mobilenetv3_large'
    model_name = 'mobilenetv3_small'
    path = './datasets'

    train_path = os.path.join(path, 'train.txt')
    test_path = os.path.join(path, 'test.txt')
    val_path = os.path.join(path, 'test.txt')

    def __init__(self):
        self.scale = 1.59
        self.per_device_batch_size = 16 if self.im_size == 224 else 4
        self.batch_size = self.per_device_batch_size * max(self.num_gpus, 1)
        self.transform_train = transforms.Compose([
            transforms.Resize((int(self.im_size * self.scale), self.im_size)),
            # transforms.RandomFlipLeftRight(),
            # Normal_Y(),
            transforms.RandomColorJitter(brightness=self.jitter_param, contrast=self.jitter_param,
                                         saturation=self.jitter_param),
            transforms.ToTensor(),
            transforms.Normalize([0.41432491, 0.41432491, 0.41432491], [0.04530748, 0.04530748, 0.04530748])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((int(self.im_size * self.scale), self.im_size)),
            # Normal_Y(),
            transforms.ToTensor(),
            transforms.Normalize([0.41432491, 0.41432491, 0.41432491], [0.04530748, 0.04530748, 0.04530748])
        ])
