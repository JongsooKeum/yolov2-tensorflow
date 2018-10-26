from learning.utils import calculate_anchor_boxes, maybe_mkdir
import os.path as osp
import argparse
import glob
import json
import numpy as np


IM_SUBDIR = 'images'
ANNO_SUBDIR = 'annotations'
AB_SUBDIR = 'anchor_boxes'


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('ds_dir', help='directory where the dataset is located')
    parser.add_argument('-p', '--pattern', help='pattern of image file names to use, default to *.png',
                        default='*.png')
    parser.add_argument('-n', '--num_samples', help='number of samples to use, default to using all',
                        default=None, type=int)
    parser.add_argument('-a', '--num_ab', help='number of anchor boxes', default=2, type=int)

    return parser.parse_args()


def _main(cfg):
    im_dir = osp.join(cfg.ds_dir, IM_SUBDIR)
    anno_dir = osp.join(cfg.ds_dir, ANNO_SUBDIR)
    ab_dir = osp.join(cfg.ds_dir, AB_SUBDIR)
    out_path = osp.join(ab_dir, 'calculated_{}.json'.format(cfg.num_ab))
    if osp.isfile(out_path):
        print('File named {} already exists.'.format(out_path))
        exit()

    maybe_mkdir(ab_dir)

    # Get list of data samples. (Sorted by name)
    im_paths = glob.glob(osp.join(im_dir, cfg.pattern))
    num_images = len(im_paths)
    print('Num images found: {}'.format(num_images))
    if num_images <= 0:
        print('No images are found.')
        return
    im_paths = np.random.permutation(im_paths)

    num_images = len(im_paths) if cfg.num_samples is None else min(len(im_paths), cfg.num_samples)
    print('{} out of {} found image files will be dealt with.'.format(num_images, len(im_paths)))
    print('{} ~ {}'.format(im_paths[0], im_paths[num_images - 1]))
    if num_images != len(im_paths):
        im_paths = im_paths[:num_images]

    # Calculate and save the anchor boxes.
    print('Calculating the anchor boxes...')
    anchors = calculate_anchor_boxes(im_paths, anno_dir, cfg.num_ab)
    with open(out_path, 'w') as f:
        json.dump(anchors.tolist(), f, indent=4)
    print('Anchor boxes are saved: {}'.format(out_path))


if __name__ == '__main__':
    args = _parse_args()
    _main(args)
