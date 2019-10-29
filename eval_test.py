import os
import torch
import PIL.Image as Image
from tqdm import tqdm
from model import Net
from data_loader import transform
from utils import load_state, load_model as ld_model


def load_model(model, checkpoint_path):
    state = load_state(checkpoint_path)
    ld_model(model, state)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def do_test(checkpoint_path, data_path, out_path):
    model = Net()
    load_model(model, checkpoint_path)
    # model.cuda()
    model.eval()

    test_dir = os.path.join(data_path, 'test_images')
    output_file = open(out_path, "w")
    output_file.write("Filename,ClassId\n")
    for f in tqdm(os.listdir(test_dir)):
        if 'ppm' in f:
            data = transform(pil_loader(test_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]

            file_id = f[0:5]
            output_file.write("%s,%d\n" % (file_id, pred))

    output_file.close()


if __name__ == '__main__':
    do_test('checkpoints/stn7/epoch_20.pth', 'data/nyucvfall2019', 'submission.csv')
