from mxnet.gluon.data.vision.datasets import dataset
from mxnet import image
import numpy as np
import os

class_id = {
    '0-0': 0,
    '0-1': 1,
    '0-2': 2,
    '1-0': 3,
    '1-1': 4,
    '1-2': 5,
    '2-0': 6,
    '2-1': 7,
    '2-2': 8
}


class ImageTxtDataset(dataset.Dataset):
    def __init__(self, root, flag=1):
        print('loading dataset...')
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self._list_images(self._root)

    def _list_images(self, root):
        self.items = []
        with open(root, 'r') as f_txt:
            for line in f_txt:
                index = line.split('/')[0]
                label_x = int(index.split('-')[0])
                label_y = int(index.split('-')[1])
                self.items.append([line, np.array((label_x, label_y, class_id[index]))])

        """
        for example in glob(os.path.join(root, '**', '*.bmp')):
            index = example.split('/')[2]
            label_x = int(index.split('-')[0])
            label_y = int(index.split('-')[1])
            # [image_path, ndarray(x, y, 9cls_id)]
            self.items.append([example, np.array((label_x, label_y, class_id[index]))])
        """

    def __getitem__(self, idx):
        """

        :param idx: image id
        :return: img: RGB, label:np.ndarray, example, array([1., 2.], dtype=float32)
        """
        # image_path
        img = image.imread(os.path.join('./datasets', self.items[idx][0].strip()), self._flag)

        # ndarray(x, y)--->(2, )
        label = self.items[idx][1]

        return img, label

    def __len__(self):
        return len(self.items)
