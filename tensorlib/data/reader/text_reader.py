import json
from tensorlib.data.reader.tfrecord import *


def read_annotation(path, keys):
    metas = []
    with open(path) as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line = json.loads(line)
            image = {'image': line['image'],
                     'height': line['height'],
                     'width': line['width']}
            ann = {key: line[key] for key in keys}
            meta = MetaData(image=image, ann=ann)
            metas.append(meta)
