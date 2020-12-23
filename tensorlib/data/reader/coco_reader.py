from pycocotools.coco import COCO
from tensorlib.data.reader.tfrecord import *
import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import random
import shutil
import json
import io


def _add_neck(joints):
    joints = joints.astype(np.float)
    left_shoulder = joints[5:6, :2]
    right_shoulder = joints[6:7, :2]
    m = joints[5:6, 2:3] * joints[6:7, 2:3]
    neck = np.concatenate([(right_shoulder + left_shoulder) / 2., m], axis=-1)
    joints = np.concatenate([joints[:1], neck, joints[1:]], axis=-2)
    return joints.astype(np.int16)


class KeyPointEncoder(object):

    @staticmethod
    def un_pack(ann):
        keypoints = np.array(ann['keypoints'], dtype=np.int16).reshape((17, 3))
        keypoints = _add_neck(keypoints)
        segmentation = ann['segmentation'].astype(np.uint8)
        x, y, w, h = tuple(ann['bbox'])
        bbox = (int(x), int(y), int(x + w + 1), int(y + h + 1))
        keypoints[:, 0] -= bbox[0]
        keypoints[:, 1] -= bbox[1]
        segmentation = segmentation[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        return keypoints, segmentation, bbox

    def __call__(self, meta):
        keypoints, segmentation, bbox = self.un_pack(meta.ann)
        h, w = segmentation.shape
        image = cv2.imread(meta.path)[bbox[1]: bbox[3], bbox[0]: bbox[2]]
        image = Image.fromarray(image)
        output_io = io.BytesIO()
        image.save(output_io, format='JPEG')
        image = output_io.getvalue()
        feature = {'image/name': bytes_feature(meta.image['file_name']),
                   'image/dataset': bytes_feature(image),
                   'image/height': int64_feature(h),
                   'image/width': int64_feature(w),
                   'segmentation': bytes_feature(segmentation.tobytes()),
                   'keypoints': bytes_feature(keypoints.tobytes())}
        return {'feature': feature}


ENCODERS = {'person_keypoints': KeyPointEncoder,
            'captions': None,
            'instances': None}


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


def func(data_dir, data_type, year=2014):
    json_dir = os.path.join(data_dir, 'json')
    train_dir = os.path.join(json_dir, 'train')
    val_dir = os.path.join(json_dir, 'val')
    test_dir = os.path.join(json_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    train_name = '{}_{}{:d}.txt'.format(data_type, 'train', year)
    val_name = '{}_{}{:d}.txt'.format(data_type, 'val', year)
    test_name = '{}_{}{:d}.txt'.format(data_type, 'test', year)
    with open(os.path.join(val_dir, val_name), 'r') as f:
        lines = f.readlines()
        train_cut = int(len(lines) * 0.8)
        val_cut = int(len(lines) * 0.9)
        random.shuffle(lines)
        with open(os.path.join(train_dir, train_name), 'a') as fw:
            for line in lines[:train_cut]:
                fw.writelines(line + '\n')
                line = json.loads(line)
                shutil.move(os.path.join(val_dir, line['image']),
                            os.path.join(train_dir, line['image']))
                shutil.move(os.path.join(val_dir, line['segmentation']),
                            os.path.join(train_dir, line['segmentation']))
        with open(os.path.join(test_dir, test_name), 'w') as fw:
            for line in lines[val_cut:]:
                fw.writelines(line + '\n')
                line = json.loads(line)
                shutil.move(os.path.join(val_dir, line['image']),
                            os.path.join(test_dir, line['image']))
                shutil.move(os.path.join(val_dir, line['segmentation']),
                            os.path.join(test_dir, line['segmentation']))
        with open(os.path.join(val_dir, val_name), 'w') as fw:
            for line in lines[train_cut: val_cut]:
                fw.writelines(line + '\n')


def read_coco(data_dir, data_type, year=2014, division='train',
              sup_nms=None, cat_nms=None, cat_ids=None):
    sup_nms = sup_nms or []
    cat_nms = cat_nms or []
    cat_ids = cat_ids or []
    path = get_ann_path(data_dir, data_type, year=year, division=division)
    coco = COCO(path)
    cat_ids = coco.getCatIds(catNms=cat_nms, supNms=sup_nms, catIds=cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)
    return coco, cat_ids, img_ids


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


def read_annotation(data_dir, data_type, year=2014, division='train',
                    sup_nms=None, cat_nms=None, cat_ids=None):
    sup_nms = sup_nms or []
    cat_nms = cat_nms or []
    cat_ids = cat_ids or []
    path = get_ann_path(data_dir, data_type, year=year, division=division)
    coco = COCO(path)
    cat_ids = coco.getCatIds(catNms=cat_nms, supNms=sup_nms, catIds=cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(img_ids)
    metas = []
    if division == 'demo':
        division = 'val'
    for img_id, img in zip(img_ids, imgs):
        ann_id = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_id)
        anns = list(filter(select, anns))
        if len(anns) == 0:
            continue
        path = os.path.join(data_dir, division + str(year), img['file_name'])
        for ann in anns:
            ann['segmentation'] = coco.annToMask(ann)
            meta = MetaData(path=path,
                            image=img, ann=ann)
            metas.append(meta)
    return metas


def generate_demo_ann(data_dir, data_type, num=100, year=2014, division='train',
                      sup_nms=None, cat_nms=None, cat_ids=None):
    coco, cat_ids, img_ids = read_coco(data_dir, data_type, year=year,
                                       division=division, sup_nms=sup_nms,
                                       cat_nms=cat_nms, cat_ids=cat_ids)
    img_ids = random.sample(img_ids, num)
    imgs = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds(imgIds=img_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    info = coco.dataset['info']
    categories = coco.loadCats(cat_ids)
    demo_coco = {'info': info, 'images': imgs,
                 'annotations': anns, 'categories': categories}
    target_path = os.path.join(data_dir, 'annotations',
                               data_type + '_demo{:d}.json'.format(year))
    json.dump(demo_coco, open(target_path, 'w'))


def demo():
    data_type = 'person_keypoints'
    data_dir = "D:/GeekGank/workspace/Data/coco"
    out_dir = "D:/GeekGank/workspace/Data/coco/tfrecord_demo"
    encoder = ENCODERS[data_type]()
    demo_metas = read_annotation(data_dir=data_dir, data_type=data_type, division='demo')
    assign_tasks(root_dir=out_dir, metas=demo_metas, encoder=encoder, num_shards=4, division='demo')


def main():
    data_type = 'person_keypoints'
    data_dir = "D:/GeekGank/workspace/Data/coco"
    out_dir = "D:/GeekGank/workspace/Data/coco/tfrecord"
    encoder = ENCODERS[data_type]()
    train_metas = read_annotation(data_dir=data_dir, data_type=data_type, division='train')
    val_metas = read_annotation(data_dir=data_dir, data_type=data_type, division='val')
    train_cutoff = int(0.8 * len(val_metas))
    val_cutoff = int(0.9 * len(val_metas))
    train_metas += val_metas[0: train_cutoff]
    val = val_metas[train_cutoff: val_cutoff]
    test = val_metas[val_cutoff:]
    assign_tasks(root_dir=out_dir, metas=train_metas, encoder=encoder, num_shards=256, division='train')
    assign_tasks(root_dir=out_dir, metas=val, encoder=encoder, num_shards=8, division='val')
    assign_tasks(root_dir=out_dir, metas=test, encoder=encoder, num_shards=8, division='test')


def pack_anns(img, anns):
    info = json.dumps(anns)
    len_str = str(len(info)).zfill(5)
    len_byte = bytes(len_str, encoding='utf-8')
    json_byte = bytes(info, encoding='utf-8')
    body = len_byte + json_byte + img
    return body


def read(data_dir, num=None):
    dat_type = 'instances'
    year = 2014
    division = 'val'
    coco, cat_ids, img_ids = read_coco(data_dir=data_dir,
                                       data_type=dat_type,
                                       year=year,
                                       division=division)
    if num is not None:
        img_ids = random.sample(img_ids, 400)
    imgs = coco.loadImgs(img_ids)
    bodies = []
    for img, img_id in zip(imgs, img_ids):
        with open(os.path.join(data_dir, division + str(year), img['file_name']), 'rb') as fi:
            img = fi.read()
        anns = coco.loadAnns(coco.getAnnIds(img_id))
        bboxes = [ann['bbox'] for ann in anns]
        category_ids = [ann['category_id'] for ann in anns]
        ids = [ann['id'] for ann in anns]
        anns = dict(bboxes=bboxes, category_ids=category_ids, ids=ids)
        body = pack_anns(img, anns)
        bodies.append(body)
    return bodies


if __name__ == '__main__':
    read_annotation_to_json('D:/GeekGank/workspace/Data/coco',
                            'person_keypoints')
