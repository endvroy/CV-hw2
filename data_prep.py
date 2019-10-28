import os
import random
from tqdm import tqdm


def reorg_ds_path(data_path):
    train_path = os.path.join(data_path, 'train_images')
    test_path = os.path.join(data_path, 'test_images')
    tmp_train_path = train_path + '.old'
    tmp_test_path = test_path + '.old'
    os.rename(train_path, tmp_train_path)
    os.rename(test_path, tmp_test_path)
    os.rename(os.path.join(tmp_train_path, 'train_images'), train_path)
    os.rename(os.path.join(tmp_test_path, 'test_images'), test_path)
    os.rmdir(tmp_train_path)
    os.rmdir(tmp_test_path)


def make_val_ds(data_path, val_ratio=0.2):
    train_path = os.path.join(data_path, 'train_images')
    val_path = os.path.join(data_path, 'val_images')
    for cat in tqdm(os.listdir(train_path)):
        train_cat_path = os.path.join(train_path, cat)
        if os.path.isdir(train_cat_path):
            val_cat_path = os.path.join(val_path, cat)  # the path to copy val data to
            os.makedirs(val_cat_path)  # make dir
            train_f_list = os.listdir(train_cat_path)  # train data in category cat
            val_n = int(len(train_f_list) * val_ratio)  # calc n of val images
            # randomly pick val files
            random.shuffle(train_f_list)
            val_fnames = train_f_list[:val_n]
            for val_fname in val_fnames:
                os.rename(os.path.join(train_cat_path, val_fname),
                          os.path.join(val_cat_path, val_fname))


if __name__ == '__main__':
    data_path = 'data/nyucvfall2019'
    # reorg_ds_path(data_path)
    make_val_ds(data_path)
