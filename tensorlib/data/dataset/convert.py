import scipy.io as io
import numpy as np
import os
import cv2
import time
import json
from tensorlib.utils import list_files

LSP_JOINTS = ['right_ankle', 'right_knee', 'right_hip',
              'left_hip', 'left_knee', 'left_ankle',
              'pelvis', 'thorax', 'right_wrist',
              'right_elbow', 'right_shoulder', 'left_shoulder',
              'left_elbow', 'left_wrist', 'neck', 'head top']

MPII_JOINTS = ['right_ankle', 'right_knee', 'right_hip',
               'left_hip', 'left_knee', 'left_ankle',
               'pelvis', 'thorax', 'neck',
               'head top', 'right_wrist', 'right_elbow',
               'right_shoulder', 'left_shoulder', 'left_elbow',
               'left_wrist']

SKELETON = [[0, 1], [1, 2], [2, 6], [8, 12], [12, 11], [11, 10],
            [5, 4], [4, 3], [3, 6], [8, 13], [13, 14], [14, 15],
            [6, 8], [8, 9]]


def add_joint(joints, left, right, index):
    joints = joints.astype(np.float)
    lj = joints[:, left:left + 1, :2]
    rj = joints[:, right:right + 1, :2]
    v = joints[:, left:left + 1, 2:3] * joints[:, right:right + 1, 2:3]
    joint = np.concatenate([(lj + rj) / 2., v], axis=2)
    joints = np.concatenate([joints[:, :index], joint, joints[:, index:]], axis=1)
    return joints.astype(np.int16)


def lsp2coco(data_dir):
    img_dir = os.path.join(data_dir, 'images')
    ann_path = os.path.join(data_dir, 'joints.mat')
    ann = io.loadmat(ann_path)
    joints = np.transpose(ann['joints'], axes=[2, 1, 0])
    # change default visible flag 0 to 1
    joints[:, :, 2] = 1 - joints[:, :, 2]
    # add pelvis
    joints = add_joint(joints, 3, 2, 6)
    # add thorax with zero and adjust neck & head top position
    joints = np.concatenate([joints[:, :7], np.zeros_like(
        joints[:, 0:1]), joints[:, -2:], joints[:, 7:-2]], axis=1)
    info = {"description": "LSP 2010 Dataset",
            "url": "http://sam.johnson.io/research/lsp.html",
            "version": "1.0", "year": 2010,
            "contributor": "Johnson, Sam and Everingham, Mark",
            "date_created": "2010/01/30"}
    imgs = []
    anns = []
    for i, path in enumerate(list_files(img_dir)):
        image = cv2.imread(path)
        mtime = os.stat(path).st_mtime
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        name = os.path.basename(path)
        img_id = int(''.join(filter(str.isdigit, name)))
        imgs.append({"license": 1, "file_name": 'images/' + name, "coco_url": None,
                     "height": image.shape[0], "width": image.shape[1],
                     "data_captured": mtime, "flickr_url": None,
                     "id": img_id})
        joint = joints[0]
        ann_kps = joint[joint[:, 2] == 1, :].reshape(-1, 3)
        xmin = np.min(ann_kps[:, 0])
        ymin = np.min(ann_kps[:, 1])
        xmax = np.max(ann_kps[:, 0])
        ymax = np.max(ann_kps[:, 1])
        width = xmax - xmin - 1
        height = ymax - ymin - 1
        # corrupted bounding box
        if width <= 0 or height <= 0:
            bbox = []  # 20% extend
        else:
            bbox = [max((xmin + xmax) / 2. - width / 2 * 1.2, 0),
                    max((ymin + ymax) / 2. - height / 2 * 1.2, 0),
                    width * 1.2, height * 1.2]
        anns.append({"segmentation": [], "num_keypoints": int(np.sum(joint[:, 2] == 1)),
                     "area": bbox[2] * bbox[3], "iscrowd": 0, "keypoints": joint.reshape(-1).tolist(),
                     "image_id": img_id, "bbox": bbox, "category_id": 1, "id": i})
    categories = [{"supercategory": "person", "name": "person",
                   "skeleton": SKELETON, "keypoints": MPII_JOINTS,
                   "id": 1}]
    coco_format = {'info': info, 'images': imgs,
                   'annotations': anns, 'categories': categories}
    target_path = os.path.join(data_dir, 'lsp', 'annotations', 'person_keypoints_train2010.json')
    json.dump(coco_format, open(target_path, 'w'), indent=2)


def check_empty(value, name):
    try:
        value[name]
    except ValueError:
        return True

    if len(value[name]) > 0:
        return False
    else:
        return True


def mpii2coco(data_dir, division='train'):
    db_type = 1 if division == 'train' else 0
    img_dir = os.path.join(data_dir, 'images')
    ann_path = os.path.join(data_dir, 'joints.mat')
    data = io.loadmat(ann_path)['RELEASE']
    info = {"description": "MPII 2014 Dataset",
            "url": "http://human-pose.mpi-inf.mpg.de",
            "version": "1.0", "year": 2014,
            "contributor": "Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt",
            "date_created": "2014/06/01"}
    aid = 0
    imgs = []
    anns = []
    for img_id in range(len(data['annolist'][0][0][0])):
        if data['img_train'][0][0][0][img_id] == db_type and \
                not check_empty(data['annolist'][0][0][0][img_id], 'annorect'):
            name = str(data['annolist'][0][0][0][img_id]['image'][0][0][0][0])
            path = os.path.join(img_dir, name)
            image = cv2.imread(path)
            mtime = os.stat(path).st_mtime
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
            imgs.append({"license": 1, "file_name": 'images/' + name, "coco_url": None,
                         "height": image.shape[0], "width": image.shape[1],
                         "data_captured": mtime, "flickr_url": None,
                         "id": img_id})
            if not db_type:
                continue
            for pid in range(len(data['annolist'][0][0][0][img_id]['annorect'][0])):
                if not check_empty(data['annolist'][0][0][0][img_id]['annorect'][0][pid],
                                   'annopoints'):  # kps is annotated
                    bbox = np.zeros([4])  # xmin, ymin, w, h
                    kps = np.zeros((len(MPII_JOINTS), 3), np.int16)  # xcoord, ycoord, vis
                    for jid in range(len(data['annolist'][0][0][0][img_id]['annorect'][
                                             0][pid]['annopoints']['point'][0][0][0])):
                        ann_jid = data['annolist'][0][0][0][img_id]['annorect'][0][pid][
                            'annopoints']['point'][0][0][0][jid]['id'][0][0]
                        kps[ann_jid][0] = data['annolist'][0][0][0][img_id]['annorect'][
                            0][pid]['annopoints']['point'][0][0][0][jid]['x'][0][0]
                        kps[ann_jid][1] = data['annolist'][0][0][0][img_id]['annorect'][
                            0][pid]['annopoints']['point'][0][0][0][jid]['y'][0][0]
                        kps[ann_jid][2] = 1
                    ann_kps = kps[kps[:, 2] == 1, :].reshape(-1, 3)
                    xmin = np.min(ann_kps[:, 0])
                    ymin = np.min(ann_kps[:, 1])
                    xmax = np.max(ann_kps[:, 0])
                    ymax = np.max(ann_kps[:, 1])
                    width = xmax - xmin - 1
                    height = ymax - ymin - 1
                    # corrupted bounding box
                    if width <= 0 or height <= 0:
                        continue
                    # 20% extend
                    else:
                        bbox[0] = np.maximum((xmin + xmax) / 2. - width / 2 * 1.2, 0)
                        bbox[1] = np.maximum((ymin + ymax) / 2. - height / 2 * 1.2, 0)
                        bbox[2] = width * 1.2
                        bbox[3] = height * 1.2
                    anns.append({"segmentation": [], "num_keypoints": int(np.sum(kps[:, 2] == 1)),
                                 "area": bbox[2] * bbox[3], "iscrowd": 0, "keypoints": kps.reshape(-1).tolist(),
                                 "image_id": img_id, "bbox": bbox.tolist(), "category_id": 1, "id": aid})
                    aid += 1

    categories = [{"supercategory": "person", "name": "person",
                   "skeleton": [[0, 1], [1, 2], [2, 6], [7, 12],
                                [12, 11], [11, 10], [5, 4], [4, 3],
                                [3, 6], [7, 13], [13, 14], [14, 15],
                                [6, 7], [7, 8], [8, 9]],
                   "keypoints": MPII_JOINTS, "id": 1}]
    coco_format = {'info': info, 'images': imgs,
                   'annotations': anns, 'categories': categories}
    target_path = os.path.join(data_dir, 'mpii', 'annotations', 'person_keypoints_{}2014.json'.format(division))
    json.dump(coco_format, open(target_path, 'w'), indent=2)
