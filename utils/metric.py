from mxnet.metric import EvalMetric
from mxnet import nd


class BinaryAccMetric(EvalMetric):
    def __init__(self, config):
        super(BinaryAccMetric, self).__init__('BinaryAcc')
        self.config = config
        self.RET = None

    def update(self, labels, preds):
        """
        :param labels: [(batch_per_device, ), (), ...]
        :param preds: [(batch_per_device, 1), (), ...]
        :return:
        """
        # binary_label = labels
        # binary_cls_logits = preds[0]
        num_acc = 0
        for lb, pd in zip(labels, preds):
            tp_RET = nd.concatenate([nd.expand_dims(lb, axis=1), pd], axis=1)
            if self.RET is None:
                self.RET = tp_RET
            else:
                self.RET = nd.concatenate([self.RET, tp_RET], axis=0)
            pred_label = nd.squeeze(pd) >= 0.5

            tp = nd.sum(pred_label == lb)
            num_acc = num_acc + tp.asscalar()

        self.sum_metric += num_acc
        self.num_inst += self.config.batch_size


class Accuracy_Y(EvalMetric):
    def __init__(self):
        super(Accuracy_Y, self).__init__('XY_Acc')

    def update(self, labels, preds):
        labels_xy = labels
        output_x, output_y = preds
        for out_x, out_y, lb_xy in zip(output_x, output_y, labels_xy):
            # --->(batch_num, )
            out_x_label = nd.argmax(out_x, axis=-1)
            out_y_label = nd.argmax(out_y, axis=-1)
            out_xy = out_x_label * 3 + out_y_label
            # lb_xy --->(batch_num, )
            out_xy = out_xy.asnumpy().astype('int32')
            lb_xy = lb_xy.asnumpy().astype('int32')

            num_correct = (out_xy == lb_xy).sum()

            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(out_xy)
            self.global_num_inst += len(out_xy)
