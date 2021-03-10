import numpy as np
import tensorflow as tf
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
import os
import cv2

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

joint_names = [
    "nose",
    "neck",
    "pelvis",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"]

Inter_Threshold = 0.05
InterMinAbove_Threshold = 6

Min_Subset_Cnt = 4
Min_Subset_Score = 0.8


def get_boxes(shape, window_size):
    shift_x = tf.cast(tf.range(0, shape[1]), tf.float32) + 0.5
    shift_y = tf.cast(tf.range(0, shape[1]), tf.float32) + 0.5
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.reshape(shift_x, [-1])
    shift_y = tf.reshape(shift_y, [-1])
    shifts = tf.stack([
        shift_x, shift_y, shift_x, shift_y], axis=0)
    half = window_size / 2
    boxes = tf.transpose(shifts) + tf.constant([-half, -half, half, half], dtype=tf.float32)
    return boxes


def non_max_suppression(tf_input, window_size, top_k=10, map_threshold=0.1, nms_threshold=0.01):
    """
    arg:
        tf_input: A tensor shape with (height, width)
    return:
    """
    shape = tf.shape(tf_input)
    boxes = get_boxes(shape, window_size=window_size)
    scores = tf.reshape(tf_input, [-1])
    mask = tf.cast(tf.greater(scores, map_threshold), scores.dtype)
    boxes = boxes * tf.expand_dims(mask, axis=-1)
    scores = scores * mask
    indices = tf.image.non_max_suppression(boxes, scores, top_k, nms_threshold)
    ys = tf.cast(indices // shape[1], scores.dtype)
    xs = tf.cast(indices % shape[1], scores.dtype)
    result = tf.stack([tf.gather(scores, indices), xs, ys], axis=-1)
    return result


def sample_match(scores):
    n, m = scores.shape
    scores = np.reshape(scores, newshape=(-1,))  # n x m
    indices = np.argsort(-scores)
    connections = []
    seen_x, seen_y = [], []
    weights = []
    for index in indices:
        y = index // n
        x = index % m
        if x in seen_x or y in seen_y:
            continue
        connections.append([x, y])
        weights.append(scores[index])
        seen_x.append(x)
        seen_y.append(y)
    return np.array(connections, "int32"), np.array(weights)  # [min(n, m), 2], [min(n, m)]


def weighted_hungarian(weights):
    def match(index):
        x_visited[index] = True
        for j in range(weights.shape[1]):
            if y_visited[j]:
                continue
            gap = x_weights[index] + y_weights[j] - weights[index, j]
            if gap == 0:
                y_visited[j] = True
                if y_match[j] == -1 or match(y_match[j]):
                    y_match[j] = index
                    return True
            elif slack[j] > gap:
                slack[j] = gap
        return False

    x_weights = np.max(weights, axis=1)
    y_weights = np.zeros([weights.shape[1]])
    y_match = np.full([weights.shape[1]], fill_value=-1, dtype="int32")
    slack = None
    x_visited = None
    y_visited = None
    count = min(*weights.shape)
    for i in range(weights.shape[0]):
        slack = np.full([weights.shape[1]], fill_value=np.inf)
        while 1 and count > 0:
            x_visited = np.zeros([weights.shape[0]], dtype=np.bool)
            y_visited = np.zeros([weights.shape[1]], dtype=np.bool)
            if match(i):
                count -= 1
                break
            thresh = np.min(slack[np.logical_not(y_visited)])
            x_weights[x_visited] -= thresh
            y_weights[y_visited] += thresh
            slack[np.logical_not(y_visited)] -= thresh
    matched_conns = []
    matched_ws = []
    for y, x in enumerate(y_match):
        matched_conns.append([x, y])
        matched_ws.append(weights[x, y])
    return np.array(matched_conns, "int32"), np.array(matched_ws)  # [min(n, m), 2], [min(n, m)]


def hungarian(connections):
    y_match = np.zeros([connections.shape[1]])

    def match(index):
        for j in range(connections.shape[1]):
            if connections[index, j] and not visited[j]:
                visited[j] = True
                if y_match[j] == 0 or match(y_match[j]):
                    y_match[j] = index
                    return True
        return False

    count = 0
    for i in range(connections.shape[0]):
        visited = np.zeros([connections.shape[1]], dtype=np.bool)
        count += int(match(i))
    return count


def joint2person(joints, joint_scores, paf_maps, skeleton, num_sample=10):
    """
    Dividing bunches of joints to person whom they belong to can be solved as problem
    of searching path in a weighted graph, but with tiny difference that branch of paths should
    be remained(due to the skeleton formation of human, e.g. neck has 3 branch connections), to
    do that, we just need to iterate each branch while tracing it as we used to do when searching
    no-branch path(that should be a recursive procedure), formally:

    1. Using adjacency matrix to store connections, to cope with 1-N connections, a nested dict
    was used as column in adj-matrix to map each id in N as key to each of multi-branch connections.
    e.g. neck -> left shoulder: 1-3, neck -> right shoulder: 1-4, neck -> pelvis: 1-2
    {3: 1-3 adj-array, 4: 1-4 adj-array, 2: 1-2 adj-array}

    2. In the same time, a adjacency array is prepared as a skeleton path, in a similar way, a nested
    list was used as item in adj-array to represent multi-branch nodes. Later, skeleton path will be
    used as index to assist searching work.
    e.g. [[1], [4, 3, 2], [10, 9], [5], [6], [7], [8], [-1], [-1], [11], [12], [13], [14], [-1], [-1]]
           0       1         2      3    4    5    6    7     8      9    10    11    12    13    14

    3. Given the first joint index i in skeleton, we can find each person skeleton path that start from
    i following the chain connections in adj-matrix, there may exists person skeleton paths that starts
    from other joint, for any other joint index in skeleton, we can use the same strategy to find new person
    skeleton path starting from that index's unvisited joints til the end.

    4. Now we have each person's joint indices from each part, by using these indices and their specific
    part id as index, we can retrieve joints and score belong to each person.

    :param joints: a list of ndarray [num joint, num person(variable), 2], generated from nms
    :param joint_scores: a list of ndarray [num joint, num person(variable)], generated from nms
    :param paf_maps: [H, W, num limb] with x, y placed on channel alternatively
    :param skeleton: list with length num limb nested with list with length 2
    :param num_sample: sample number for integral
    :return: joints [max person num, num joint, 3], scores [max person num]
    """
    if max(len(joint) for joint in joints) <= 1:
        joints = [np.hstack((joint, np.full([1, 1], 2, dtype=joints[0].dtype)))[None, :, :]
                  if len(joint) == 1 else np.zeros(
            [1, 1, 3], dtype=joints[0].dtype) for joint in joints]
        joints = np.hstack(joints)
        joint_scores = sum(score[0] for score in joint_scores if len(score) == 1)
        joint_scores = np.array([joint_scores])
        return joints, joint_scores
    paf_indices = list(range(len(skeleton) * 2))
    paf_indices = list(zip(paf_indices[0::2], paf_indices[1::2]))
    skeleton_path = [[-1] for _ in range(len(joints))]
    paths = [{} for _ in range(len(joints))]  # [{}] * len(joints) share same dict
    for (idx1, idx2), (paf_idx1, paf_idx2) in zip(skeleton, paf_indices):
        if skeleton_path[idx1][0] == -1:
            skeleton_path[idx1][0] = idx2
        else:
            skeleton_path[idx1].append(idx2)
        point1, point2 = joints[idx1], joints[idx2]
        paths[idx1][idx2] = np.full([len(joints)], -1, np.int8)
        if len(point1) == 0 or len(point2) == 0:
            joint_scores[idx1][:] = joint_scores[idx2][:] = 0
            continue
        # print('#'*4, joint_names[idx1], joint_names[idx2], '#'*4)
        p1_idx, p2_idx = bipartite(
            point1=point1, point2=point2,
            paf_x=paf_maps[..., paf_idx1],
            paf_y=paf_maps[..., paf_idx2],
            num_sample=num_sample)
        paths[idx1][idx2][p1_idx] = p2_idx
    return human_joints(skeleton[0][0], joints, joint_scores, paths, skeleton_path)


def bipartite(point1, point2, paf_x, paf_y, num_sample=10):
    point1 = np.expand_dims(point1, axis=1)  # [n, 1, 2]
    point1 = np.repeat(point1, repeats=point2.shape[0], axis=1)  # [n, m, 2]
    point2 = np.expand_dims(point2, axis=0)  # [1, m, 2]
    vectors = point2 - point1  # [n, m, 2], limb vectors
    norm = np.sqrt(np.sum(np.square(vectors), axis=-1))  # [n, m]
    norm[norm == 0.] = 1e-7
    unit_vectors = vectors / norm[:, :, None]  # limb unit vectors
    # Sample points in limb and do integral
    interval = 1. / num_sample
    score = np.zeros_like(norm)
    criterion1 = np.zeros_like(norm, dtype=np.int8)  # valid sample score count
    for i in range(num_sample):
        sample_point = (1. - interval * i) * point1 + interval * i * point2 + 0.5
        sample_point = sample_point.astype(np.int32)
        paf_x_v = paf_x[sample_point[:, :, 1], sample_point[:, :, 0]]  # [n, m]
        paf_y_v = paf_y[sample_point[:, :, 1], sample_point[:, :, 0]]
        sample_score = paf_x_v * unit_vectors[:, :, 0] \
            + paf_y_v * unit_vectors[:, :, 1]  # inner product [n, m]
        criterion1 += np.greater(sample_score, Inter_Threshold).astype(np.int8)
        score += sample_score
    row_idx, col_idx = linear_sum_assignment(-score)  # [min(n, m)]
    # post-selection for connections in case of overmatching
    criterion1 = criterion1[row_idx, col_idx] >= 0.5 * num_sample  # thresh: 0.5
    # suppress long connections
    dist_prior = score / num_sample + np.minimum(0.5 * paf_x.shape[1] / norm - 1, 0.)
    criterion2 = dist_prior[row_idx, col_idx] > 0
    criterion = np.logical_and(criterion1, criterion2)
    row_idx = row_idx[criterion]
    col_idx = col_idx[criterion]
    return row_idx, col_idx


def single_path(paths, skeleton_path, body, iex, idx):
    if idx == -1 or iex == -1:
        return
    body[iex] = idx
    for i in skeleton_path[iex]:
        if i == -1:
            continue
        next_idx = paths[iex][i][idx]
        single_path(paths, skeleton_path, body, i, next_idx)


def search_path(paths, skeleton_path, iex, indices):
    connections = []
    for idx in indices:
        body = np.full([len(paths)], -1, np.int8)
        single_path(paths, skeleton_path, body, iex, idx)
        connections.append(body)
    return connections


def human_joints(start, joints, joint_scores, paths, skeleton_path):
    # Find paths start from start point
    connections = search_path(paths, skeleton_path, start,
                              np.where(paths[start][skeleton_path[start][0]] != -1)[0])
    # Find paths not start from start point
    for i in range(1, len(paths)):
        cur_indices = set(body[i] for body in connections if body[i] != -1)
        full_indices = set(range(len(joints[i])))
        # new person starts from missing part in full_indices
        # (e.g.) joint i, cur_indices: [0, 1, 2] full_indices: [0, 1, 2, 3]
        # that means there exists new person starts from 3th point of joint i
        if len(cur_indices) != len(full_indices):
            extra_conns = search_path(paths, skeleton_path, i, full_indices.difference(cur_indices))
            connections += extra_conns
    humans = np.zeros([len(connections), len(joints), 3], joints[0].dtype)
    scores = np.zeros([len(connections), len(joints)], joint_scores[0].dtype)
    for i, body in enumerate(connections):
        for j, idx in enumerate(body):
            if idx != -1:
                humans[i, j][-1] = 2
                humans[i, j][:-1] = joints[j][idx]
                scores[i, j] = joint_scores[j][idx]
    scores = np.sum(scores, axis=1) / (np.count_nonzero(scores, axis=1) + 1e-7)
    return humans, scores


def resize_point(point):
    return int((8 / 2 - 0.5) + point * 8)


def draw_skeletons(image, joints, skeleton):
    for person in joints:
        for k, (i, j) in enumerate(skeleton):
            if person[i][-1] != 0:
                p1 = (resize_point(person[i][0]), resize_point(person[i][1]))
                cv2.circle(image, p1, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
            if person[j][-1] != 0:
                p2 = (resize_point(person[j][0]), resize_point(person[j][1]))
                cv2.circle(image, p2, 3, CocoColors[j], thickness=3, lineType=8, shift=0)
            if person[i][-1] != 0 and person[j][-1] != 0:
                cv2.line(image, p1, p2, CocoColors[k], 3)
    return image


def resize_joint(img_size, tar_size, joints):
    joints = joints.astype(np.float32)
    r_h, r_w = img_size
    t_h, t_w = tar_size
    ratio_h = t_h / r_h
    ratio_w = t_w / r_w
    joints[..., 0] = joints[..., 0] * ratio_w + 0.5
    joints[..., 1] = joints[..., 1] * ratio_h + 0.5
    return np.floor(joints).astype(np.int32)


def to_coco(image_id, image_size, tar_size, joints, scores):
    joints = resize_joint(image_size, tar_size, joints)
    result = []
    visible = np.full([joints.shape[1], 1], 1)
    for i in range(joints.shape[1]):
        person_joints = np.concatenate((joints[:, i, :], visible), axis=-1)
        result.append({'image_id': image_id,
                       'category_id': 1,
                       'keypoints': person_joints.reshape([-1]),
                       'score': scores[i]})
    import json
    json.dump(result, open('./eval_result.json', 'w'))


def get_ann_path(data_dir: str,
                 data_type: str,
                 year=2014,
                 division='train',
                 dataset='coco'):
    if not os.path.exists(data_dir):
        raise ValueError("Dir {} not found".format(data_dir))
    assert dataset in ['coco', 'lsp', 'mpii']
    assert data_type in ['person_keypoints', 'captions', 'instances', 'image_info']
    assert division in ['train', 'val', 'test', 'demo']
    path = '{}/{}/annotations/{}_{}{:d}.json'.format(data_dir, dataset, data_type, division, year)
    if not os.path.exists(path):
        raise ValueError("File {} not found".format(path))
    return path


def coco_eval(data_dir: str,
              data_type: str,
              eval_path: str,
              year=2014,
              division='val',
              dataset='coco'):
    cocoGt = COCO(get_ann_path(data_dir, data_type, year=year,
                               division=division, dataset=dataset))
    cocoDt = cocoGt.loadRes(eval_path)
    imgIds = list(sorted(cocoDt.getImgIds()))
    cocoEval = COCOeval(cocoGt, cocoDt, data_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def evaluate(batch_joints, batch_paf_maps, skeleton):
    results = []
    for joints, paf_maps in zip(batch_joints, batch_paf_maps):
        valid_joints = []
        valid_scores = []
        for parts in joints:
            coordinates, scores = [], []
            for part in parts:
                if part[0] > 0.01:
                    coordinates.append(part[1:].tolist())
                    scores.append(part[0].tolist())
            valid_joints.append(np.array(coordinates))
            valid_scores.append(np.array(scores))
        results.append(joint2person(valid_joints, valid_scores,
                                    paf_maps, skeleton=skeleton))
    return results
