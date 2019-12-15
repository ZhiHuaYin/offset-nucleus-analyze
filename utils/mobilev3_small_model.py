from gluoncv.model_zoo.mobilenetv3 import _MobileNetV3
from mxnet.gluon.nn import BatchNorm
import mxnet as mx
from mxnet.gluon import nn


class _MobileNetV3(_MobileNetV3):
    def __init__(self, cfg, cls_ch_squeeze, cls_ch_expand, classes):
        super(_MobileNetV3, self).__init__(cfg=cfg, cls_ch_squeeze=cls_ch_squeeze, cls_ch_expand=cls_ch_expand,
                                             multiplier=1., classes=1000, norm_kwargs=None, last_gamma=False,
                                             final_drop=0.2, use_global_stats=False, name_prefix='',
                                             norm_layer=BatchNorm)
        self.classes = classes
        with self.name_scope():
            self.output = nn.HybridSequential(prefix='output1_')
            self.output.add(
                nn.Conv2D(in_channels=cls_ch_expand, channels=self.classes,
                          kernel_size=1, prefix='fc1_'),
                nn.Flatten()
            )
            self.output1 = nn.HybridSequential(prefix='output2_')
            self.output1.add(
                nn.Conv2D(in_channels=cls_ch_expand, channels=self.classes,
                          kernel_size=1, prefix='fc2_'),
                nn.Flatten()
            )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        pred_x = self.output(x)
        pred_y = self.output1(x)
        # reutnr: X--->(batch, 3), Y--->(batch, 3)
        return F.softmax(pred_x, axis=-1), F.softmax(pred_y, axis=-1)


def get_mobilenet_v3_Y(classes=3, **kwargs):
    cfg = [
        # k, exp, c,  se,     nl,  s,
        [3, 16, 16, True, 'relu', 2],
        [3, 72, 24, False, 'relu', 2],
        [3, 88, 24, False, 'relu', 1],
        [5, 96, 40, True, 'hard_swish', 2],
        [5, 240, 40, True, 'hard_swish', 1],
        [5, 240, 40, True, 'hard_swish', 1],
        [5, 120, 48, True, 'hard_swish', 1],
        [5, 144, 48, True, 'hard_swish', 1],
        [5, 288, 96, True, 'hard_swish', 2],
        [5, 576, 96, True, 'hard_swish', 1],
        [5, 576, 96, True, 'hard_swish', 1],
    ]
    cls_ch_squeeze = 576
    cls_ch_expand = 1280

    net = _MobileNetV3(cfg, cls_ch_squeeze, cls_ch_expand, classes, **kwargs)

    return net
