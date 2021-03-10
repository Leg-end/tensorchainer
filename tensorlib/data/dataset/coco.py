from tensorlib.data.dataset.convert import *
from tensorlib.data.dataset.tfrecord import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def to_coco(dataset,
            data_dir,
            division='train'):
    if dataset == 'lsp':
        lsp2coco(data_dir)
    elif dataset == 'mpii':
        mpii2coco(data_dir, division=division)
    else:
        raise ValueError("Unknown dataset %s to convert" % dataset)


def to_tfrecord(data_dir: str,
                data_type: str,
                encoder: Encoder,
                num_shards=256,
                year=2014,
                division='train',
                dataset='coco'):
    out_dir = os.path.join(data_dir, 'tfrecord')
    metas = read_annotation(data_dir=data_dir, data_type=data_type,
                            year=year, division=division, dataset=dataset)
    assign_tasks(root_dir=out_dir, metas=metas, encoder=encoder,
                 num_shards=num_shards, division=division)


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


def read_coco(data_dir,
              data_type,
              year=2014,
              division='train',
              dataset='coco',
              sup_nms=None,
              cat_nms=None,
              cat_ids=None):
    sup_nms = sup_nms or []
    cat_nms = cat_nms or []
    cat_ids = cat_ids or []
    path = get_ann_path(data_dir, data_type, year=year,
                        division=division, dataset=dataset)
    coco = COCO(path)
    cat_ids = coco.getCatIds(catNms=cat_nms, supNms=sup_nms, catIds=cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)
    return coco, cat_ids, img_ids


def _select(ann):
    bbox = ann['bbox']
    return ann['category_id'] == 1 and ann['num_keypoints'] > 5 \
        and ann['area'] > 1600. and ann['iscrowd'] == 0 and \
        bbox[2] > 60 and bbox[3] > 60


def read_annotation(data_dir,
                    data_type,
                    year=2014,
                    division='train',
                    dataset='coco',
                    sup_nms=None,
                    cat_nms=None,
                    cat_ids=None):
    coco, cat_ids, img_ids = read_coco(
        data_dir=data_dir, data_type=data_type,
        year=year, division=division, dataset=dataset,
        sup_nms=sup_nms, cat_nms=cat_nms, cat_ids=cat_ids)
    imgs = coco.loadImgs(img_ids)
    metas = []
    if division == 'demo':
        division = 'val'
    for img_id, img in zip(img_ids, imgs):
        ann_id = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_id)
        anns = list(filter(_select, anns))
        if len(anns) == 0:
            continue
        path = os.path.join(data_dir, division + str(year), img['file_name'])
        for ann in anns:
            ann['segmentation'] = coco.annToMask(ann)
            meta = MetaData(path=path,
                            image=img, ann=ann)
            metas.append(meta)
    return metas


def gen_record(data_dir,
               data_type,
               year=2014,
               division='train',
               dataset='coco',
               sup_nms=None,
               cat_nms=None,
               cat_ids=None):
    coco, cat_ids, img_ids = read_coco(
        data_dir=data_dir, data_type=data_type,
        year=year, division=division, dataset=dataset,
        sup_nms=sup_nms, cat_nms=cat_nms, cat_ids=cat_ids)
    imgs = coco.loadImgs(img_ids)
    for img, img_id in zip(imgs, img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        # get annotations of per image
        anns = coco.loadAnns(ann_ids)
        # anns = list(filter(_select, anns))
        yield img, anns


def demo_sample(data_dir,
                data_type,
                num=100,
                year=2014,
                division='train',
                dataset='coco',
                sup_nms=None,
                cat_nms=None,
                cat_ids=None):
    coco, cat_ids, img_ids = read_coco(data_dir, data_type, year=year,
                                       division=division, dataset=dataset,
                                       sup_nms=sup_nms, cat_nms=cat_nms,
                                       cat_ids=cat_ids)
    img_ids = random.sample(img_ids, num)
    imgs = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds(imgIds=img_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    info = coco.dataset['info']
    categories = coco.loadCats(cat_ids)
    demo_coco = {'info': info, 'images': imgs,
                 'annotations': anns, 'categories': categories}
    target_path = os.path.join(data_dir, dataset, 'annotations',
                               data_type + '_demo{:d}.json'.format(year))
    json.dump(demo_coco, open(target_path, 'w'), indent=2)


def evaluation(data_dir: str,
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


def remedy(data_dir: str,
           data_type: str,
           year=2014,
           division='val',
           dataset='coco'):
    new_path = os.path.join(data_dir, dataset, data_type + '_' + division + str(year))
    coco, cat_ids, img_ids = read_coco(data_dir, data_type, year=year,
                                       division=division, dataset=dataset)
    imgs = coco.loadImgs(img_ids)
    ann_ids = coco.getAnnIds(imgIds=img_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        bbox = ann['bbox']
        if bbox[0] < 0.:
            bbox[2] += bbox[0]
            bbox[0] = 0.
        if bbox[1] < 0.:
            bbox[3] += bbox[1]
            bbox[1] = 0.
    info = coco.dataset['info']
    categories = coco.loadCats(cat_ids)
    demo_coco = {'info': info, 'images': imgs,
                 'annotations': anns, 'categories': categories}
    json.dump(demo_coco, open(new_path + '.json', 'w'), indent=2)


if __name__ == '__main__':
    remedy(data_dir="D:/GeekGank/workspace/Data/HUMAN",
           data_type="person_keypoints",
           year=2014,
           dataset='mpii',
           division='train')
