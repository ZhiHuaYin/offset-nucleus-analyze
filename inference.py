from mxnet import gluon
import mxnet as mx
from gluoncv.data.transforms import image as timage

from mxnet import image, nd
import os


class DATATransformer(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=224, scale=1.592):
        self._mean = mean
        self._std = std
        self._size = size
        self.scale = scale

    def __call__(self, im):
        im = timage.imresize(mx.nd.array(im), int(self._size * self.scale), self._size)
        im = mx.nd.image.to_tensor(im)
        im = mx.nd.image.normalize(im, mean=self._mean, std=self._std)
        return im


def transform(im, data_trans):
    im = data_trans(im)
    im = nd.expand_dims(im, axis=0)
    return im


if __name__ == '__main__':
    """
    import numpy as np

    ret = np.load('./work-dir/6.npy')
    pass
    """
    json = './work-dir/model_x-0.9739_y-0.9576_xy-0.9315-symbol.json'
    params = './work-dir/model_x-0.9739_y-0.9576_xy-0.9315-0021.params'

    ROOT = './datasets/'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im_path = os.path.join(ROOT, '1-0/1-0_new/00000_20191206_105040_0771_0038.bmp')
    im = image.imread(im_path)
    data_trans = DATATransformer(mean=mean, std=std, size=224)

    ctx = mx.gpu()
    # prefix = model_0.8307-
    net = gluon.SymbolBlock.imports(json, ['data'], params, ctx=ctx)

    x = transform(im, data_trans)

    ret = net(x.copyto(ctx))

    print(ret)

