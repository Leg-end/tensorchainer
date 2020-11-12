import numpy as np
import cv2
import random
from tensorlib.contrib.data_flow.augmenter import Aug


class RandomFlip(Aug):
    def __init__(self, prob, backend=None):
        super(RandomFlip, self).__init__(backend=backend)
        self.prob = prob

    def augment(self, record):
        if self.prob < random.uniform(0, 1.0):
            return record

        if record.image is None:
            raise ValueError("")

        height, width, _ = np.shape(record.image)
        record.image = cv2.flip(record.image, 1)

        if record.segment is not None:
            record.segment = cv2.flip(record.segment, 1)

        if record.key_points is not None:
            flip_list = (0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16)
            new_joints = []
            for people in record.key_points:
                new_keypoints = []
                for k in flip_list:
                    point = people[k]
                    if point[0] < 0 or point[1] < 0:
                        new_keypoints.append((0, 0, 0))
                        continue
                    if point[0] > width - 1 or point[1] > height - 1:
                        new_keypoints.append((0, 0, 0))
                        continue
                    if (width - point[0]) > width - 1:
                        new_keypoints.append((0, 0, 0))
                        continue
                    new_keypoints.append((width - point[0], point[1], point[2]))
                new_joints.append(new_keypoints)
            record.key_points = np.asarray(new_joints, dtype=np.int64)
        return record
