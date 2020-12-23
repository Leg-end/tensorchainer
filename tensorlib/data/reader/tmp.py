from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import io
import os
import cv2
import json


def _add_neck(joints):
    joints = joints.astype(np.float)
    left_shoulder = joints[5:6, :2]
    right_shoulder = joints[6:7, :2]
    m = joints[5:6, 2:3] * joints[6:7, 2:3]
    neck = np.concatenate([(right_shoulder + left_shoulder) / 2., m], axis=-1)
    joints = np.concatenate([joints[:1], neck, joints[1:]], axis=-2)
    return joints.astype(np.int16)


def get_ann_path(data_dir: str, data_type: str, year=2014, division='train'):
    if not os.path.exists(data_dir):
        raise ValueError("Dir {} not found".format(data_dir))
    assert data_type in ['person_keypoints', 'captions', 'instances', 'image_info']
    assert division in ['train', 'val', 'test', 'demo']
    path = '{}/annotations/{}_{}{:d}.json'.format(data_dir, data_type, division, year)
    if not os.path.exists(path):
        raise ValueError("File {} not found".format(path))
    return path


def select(ann):
    bbox = ann['bbox']
    return ann['category_id'] == 1 and ann['num_keypoints'] > 5 \
        and ann['area'] > 1600. and ann['iscrowd'] == 0 and \
        bbox[2] > 60 and bbox[3] > 60


def clip_image(image, ann):
    x, y, w, h = tuple(ann['bbox'])
    bbox = (int(x), int(y), int(x + w + 1), int(y + h + 1))
    image = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]
    output_io = io.BytesIO()
    image = Image.fromarray(image)
    image.save(output_io, format='JPEG')
    image = output_io.getvalue()
    return image, bbox


def read_annotation_to_json(data_dir, data_type, year=2014, division='train',
                            sup_nms=None, cat_nms=None, cat_ids=None):
    sup_nms = sup_nms or []  # super category name
    cat_nms = cat_nms or []  # category name
    cat_ids = cat_ids or []  # category id
    path = get_ann_path(data_dir, data_type, year=year, division=division)
    coco = COCO(path)
    # get category ids of specific supcat-cat's
    cat_ids = coco.getCatIds(catNms=cat_nms, supNms=sup_nms, catIds=cat_ids)
    # get image ids of these category
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(img_ids)
    if division == 'demo':
        division = 'val'
    json_dir = os.path.join(data_dir, 'json')
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    division_dir = os.path.join(json_dir, division)
    if not os.path.exists(division_dir):
        os.mkdir(division_dir)
    # store in dir data_dir/json/division
    name = '{}_{}{:d}_1.txt'.format(data_type, division, year)
    print("Convert {} dataset".format(division))
    with open(os.path.join(division_dir, name), 'w') as f:
        for img_id, img in tqdm(zip(img_ids, imgs), desc='converting...'):
            ann_id = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            # get annotations of per image
            anns = coco.loadAnns(ann_id)
            anns = list(filter(select, anns))
            if len(anns) == 0:
                continue
            path = os.path.join(data_dir, division + str(year), img['file_name'])
            image = cv2.imread(path)
            for i, ann in enumerate(anns):  # for each annotation
                num = ann['id']
                bbox = ann['bbox']  # bounding box
                bbox = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2] + 1), int(bbox[1] + bbox[3] + 1)]
                seg = coco.annToMask(ann).astype(np.uint8)[bbox[1]: bbox[3], bbox[0]: bbox[2]]  # mask to segment
                h, w = seg.shape
                clip_img = image[bbox[1]: bbox[3], bbox[0]: bbox[2]]  # clip image by bounding box
                assert seg.shape == clip_img.shape[:-1], "{} vs {}".format(str(seg.shape), str(clip_img.shape[:-1]))
                img_path = os.path.join(division_dir, '{:d}.jpg'.format(num))
                seg_path = os.path.join(division_dir, '{:d}.png'.format(num))
                cv2.imwrite(img_path, clip_img)  # save clipped image
                cv2.imwrite(seg_path, seg)  # save segment corresponding to clipped part
                keypoints = np.array(ann['keypoints'], np.int16).reshape(17, 3)
                keypoints = _add_neck(keypoints)
                # adjust keypoint points to fit clipped part
                keypoints[:, 0] = np.maximum(keypoints[:, 0] - bbox[0], 0)
                keypoints[:, 1] = np.maximum(keypoints[:, 1] - bbox[1], 0)
                meta = OrderedDict([('image', '{:d}.jpg'.format(num)), ('segmentation', '{:d}.png'.format(num)),
                                    ('height', h), ('width', w), ('keypoints', keypoints.tolist())])
                # meta format {"image": "1209372.jpg", "segmentation": "1209372.png", "height": 278, "width": 157,
                # "keypoints": [[85, 67, 2], [96, 62, 4], [91, 62, 2], [81, 62, 2], [105, 52, 2], [-250, -67, 0],
                # [121, 72, 2], [72, 52, 2], [142, 88, 2], [47, 40, 2], [144, 93, 2], [22, 15, 2], [101, 140, 2],
                # [71, 122, 2], [97, 194, 2], [42, 144, 2], [98, 259, 2], [27, 207, 2]]}
                meta = json.dumps(meta)
                f.writelines(meta + '\n')
