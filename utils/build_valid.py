import tqdm
import os
import numpy as np
from data.coco_pose import ref
import pickle

# this is to make pkl validation file

def build(idxes, ref):
    path, gts, info = [], [], []
    for idx in tqdm.tqdm( idxes, total = len(idxes) ):
        # collect image information and return it for making pkl file
        ann_ids = ref.coco.getAnnIds(imgIds = idx)
        ann = ref.coco.loadAnns(ann_ids)
        gts.append(ann)
        img_info = ref.coco.loadImgs(idx)[0]
        _path = img_info['file_name']
        path.append(os.path.join(ref.data_dir, _path))
        assert os.path.exists(path[-1])
        info.append(img_info)
    return {
        'path': path,
        'anns': gts,
        'idxes': idxes,
        'info': info
    }

def main():
    ref.init()
    # open valid_id file in data/coco_pose directory, and make a pkl file including image path,annotation,index
    with open(ref.ref_dir + '/valid_id', 'r') as f:
        valid_id = list(map(lambda x:int(x.strip()), f.readlines()))
    pickle.dump(build(valid_id, ref), open(ref.ref_dir + '/validation.pkl', 'wb'))

if __name__ == '__main__':
    main()
