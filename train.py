import os
import time

from utils.dataset import ImageTxtDataset
from mxnet import gluon, init
from mxnet import autograd as ag
# from utils.metric import BinaryAccMetric
import mxnet as mx
import numpy as np


def get_dataset(config):
    train_data = gluon.data.DataLoader(
        ImageTxtDataset(config.train_path).transform_first(config.transform_train),
        batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    val_data = gluon.data.DataLoader(
        ImageTxtDataset(config.val_path).transform_first(config.transform_test),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_data = gluon.data.DataLoader(
        ImageTxtDataset(config.test_path).transform_first(config.transform_test),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_data, test_data, val_data


################################################################################
"""
def analyze(data, threshold=0.5):
    # npy--->(num, 2)
    labels = data[:, 0]
    # preds is from sigmoid
    preds = data[:, 1]
    preds = np.where(preds >= threshold, 1, 0)

    P_num = np.sum(labels)
    N_num = labels.size - P_num

    index = np.where(preds == 1)
    TP = np.sum(preds[index] == labels[index]) / P_num
    # FP =
    index = np.where(preds == 0)
    TN = np.sum(preds[index] == labels[index]) / N_num
    # FN =
    return TP, TN
"""


def validate(net, val_data, ctx, epoch, config):
    metric_x = mx.metric.Accuracy()
    metric_y = mx.metric.Accuracy()
    from utils.metric import Accuracy_Y
    metric_xy = Accuracy_Y()
    # metric = BinaryAccMetric(config)
    for i, batch in enumerate(val_data):
        # []
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        # [(batch, 3), (batch, 3)...]
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        # net(X)--->return (batch, 3), (batch, 3)
        outputs = [net(X) for X in data]

        # out[0]--> (batch, 3)
        # output_? --> [(batch, 3), (batch, 3), ...]
        # 3 * x + y = 9cls_id
        output_x = [out[0] for out in outputs]
        output_y = [out[1] for out in outputs]

        # lb[0]--> (batch, )
        # labels_? --> [(batch,), ...]
        labels_x = [lb[:, 0] for lb in label]
        labels_y = [lb[:, 1] for lb in label]
        labels_xy = [lb[:, 2] for lb in label]

        metric_x.update(labels=labels_x, preds=output_x)
        metric_y.update(labels=labels_y, preds=output_y)
        metric_xy.update(labels=labels_xy, preds=(output_x, output_y))

    path = os.path.join('./OUTPUT', config.model_name, str(config.im_size))
    if not os.path.exists(path):
        os.makedirs(path)

    """
    # RET-->(sample_num, 2)
    RET = metric.RET.asnumpy()
    # save RET
    npy_path = os.path.join('./OUTPUT', config.model_name, str(config.im_size), 'npy')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    np.save('{}/{}.npy'.format(npy_path, epoch), RET)

    # TP, TN = analyze(RET, threshold=0.5)
    """

    _, val_acc_x = metric_x.get()
    _, val_acc_y = metric_y.get()
    _, val_acc_xy = metric_xy.get()
    prefix = os.path.join(path, 'model')
    net.export(path='{}_x-{:.4f}_y-{:.4f}_xy-{:.4f}'.format(prefix, val_acc_x, val_acc_y, val_acc_xy), epoch=epoch)

    # return metric.get(), TP, TN
    return metric_x.get(), metric_y.get(), metric_xy.get()


def train(finetune_net, train_data, test_data, val_data, config):
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': config.lr, 'momentum': config.momentum, 'wd': config.wd})
    metric_x = mx.metric.Accuracy()
    metric_y = mx.metric.Accuracy()
    from utils.metric import Accuracy_Y
    metric_xy = Accuracy_Y()
    Lx = gluon.loss.SoftmaxCrossEntropyLoss()
    Ly = gluon.loss.SoftmaxCrossEntropyLoss()

    # metric = BinaryAccMetric(config)
    # L = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    ################################################################################
    # Training Loop

    lr_counter = 0
    num_batch = len(train_data)

    for epoch in range(config.epochs):
        if epoch == config.lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * config.lr_factor)
            print('set learning rate to: {}'.format(trainer.learning_rate))
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric_x.reset()
        metric_y.reset()
        metric_xy.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=config.ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=config.ctx, batch_axis=0, even_split=False)
            with ag.record():
                # (batch_num, 2) --->x,y
                # net(X) - -->return (batch, 3), (batch, 3)
                outputs = [finetune_net(X) for X in data]

                # output_? -->  [(batch, 3), (batch, 3)]
                output_x = [out[0] for out in outputs]
                output_y = [out[1] for out in outputs]

                # lb[0]--> (batch, )
                # labels_? --> [(batch, ), ...]
                labels_x = [lb[:, 0] for lb in label]
                labels_y = [lb[:, 1] for lb in label]
                labels_xy = [lb[:, 2] for lb in label]

                loss_x = [Lx(yhat, y) for yhat, y in zip(output_x, labels_x)]
                loss_y = [Ly(yhat, y) for yhat, y in zip(output_y, labels_y)]

                loss = [lx + ly for lx, ly in zip(loss_x, loss_y)]

            for l in loss:
                l.backward()

            metric_x.update(labels=labels_x, preds=output_x)
            metric_y.update(labels=labels_y, preds=output_y)
            metric_xy.update(labels=labels_xy, preds=(output_x, output_y))

            trainer.step(config.batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            if i % 100 == 0:
                print('{}-th batch/Epoch:{} ing...'.format(i, epoch))

        _, train_acc_x = metric_x.get()
        _, train_acc_y = metric_y.get()
        _, train_acc_xy = metric_xy.get()

        train_loss /= num_batch

        (_, val_acc_x), (_, val_acc_y), (_, val_acc_xy) = validate(finetune_net, val_data, config.ctx, epoch, config)

        # print('TP:{}, TN:{}'.format(TP, TN))
        print('[Epoch %d] Train-acc_x: %.3f, Train-acc_y: %.3f, Train-acc_xy: %.3f,'
              'loss: %.3f | Val-acc_x: %.3f, Val-acc_y: %.3f , Val-acc-xy: %.3f'
              '| time: %.1f' % (
                  epoch, train_acc_x, train_acc_y, train_acc_xy, train_loss, val_acc_x, val_acc_y, val_acc_xy, time.time() - tic))

    (_, test_acc_x), (_, test_acc_y), (_, test_acc_xy) = validate(finetune_net, test_data, config.ctx, 99, config)
    print('[Finished] Test-acc_x: %.3f, Test-acc_y: %.3f, , Test-acc_xy: %.3f' % (test_acc_x, test_acc_y, test_acc_xy))


if __name__ == '__main__':
    from utils.config import Config

    config = Config()

    # create model
    model_name = config.model_name
    print('loading model ', model_name)
    from utils.mobilev3_small_model import get_mobilenet_v3_Y

    finetune_net = get_mobilenet_v3_Y(classes=3)

    # load params
    from mxnet import ndarray

    pretrained_path = './models/mobilenetv3_small-33c100a7.params'
    loaded = ndarray.load(pretrained_path)
    params = finetune_net._collect_params_with_prefix()
    for name in loaded:
        if 'output' not in name and name in params:
            params[name]._load_init(loaded[name], ctx=mx.cpu(), cast_dtype=False, dtype_source='current')

    """
    for param in finetune_net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    """
    finetune_net.output.initialize(init.Xavier(), ctx=config.ctx)
    finetune_net.output1.initialize(init.Xavier(), ctx=config.ctx)

    # ==================init====================
    finetune_net.collect_params().reset_ctx(config.ctx)
    finetune_net.hybridize()

    # load data
    train_data, test_data, val_data = get_dataset(config=config)

    # train
    train(finetune_net, train_data=train_data, test_data=test_data, val_data=val_data, config=config)
