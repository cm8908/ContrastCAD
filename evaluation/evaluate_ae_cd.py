from collections import defaultdict
import os
import glob
import pickle
import h5py
import numpy as np
import argparse
from joblib import Parallel, delayed
import random
import multiprocessing as mp
from scipy.spatial import cKDTree as KDTree
import time
import sys
sys.path.append("..")
from utils import read_ply
from tqdm import tqdm
from cadlib.visualize import vec2CADsolid, CADsolid2pc


PC_ROOT = "../../datasets/cad_data/pc_cad"
# data that is unable to process
SKIP_DATA = [""]

def chamfer_dist(gt_points, gen_points, offset=0, scale=1):
    gen_points = gen_points / scale - offset

    # one direction
    gen_points_kd_tree = KDTree(gen_points)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points


def process_one(path):
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(float)
        # gt_vec = fp["gt_vec"][:].astype(float)

    data_id = path.split('/')[-1].split('.')[0][:8]
    truck_id = data_id[:4]
    gt_pc_path = os.path.join(PC_ROOT, truck_id, data_id + '.ply')
    if not os.path.exists(gt_pc_path):
        return None

    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None

    if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    gt_pc = read_ply(gt_pc_path)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), args.n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    return cd

def process_one_sl(path):
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(float)
        # gt_vec = fp["gt_vec"][:].astype(float)

    data_id = path.split('/')[-1].split('.')[0][:8]
    truck_id = data_id[:4]
    gt_pc_path = os.path.join(PC_ROOT, truck_id, data_id + '.ply')
    if not os.path.exists(gt_pc_path):
        return None, None

    try:
        shape = vec2CADsolid(out_vec)
    except Exception as e:
        print("create_CAD failed", data_id)
        return None, None
    
    try:
        out_pc = CADsolid2pc(shape, args.n_points, data_id)
    except Exception as e:
        print("convert pc failed:", data_id)
        return None, None

    if np.max(np.abs(out_pc)) > 2: # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    gt_pc = read_ply(gt_pc_path)
    sample_idx = random.sample(list(range(gt_pc.shape[0])), args.n_points)
    gt_pc = gt_pc[sample_idx]

    cd = chamfer_dist(gt_pc, out_pc)
    seq_len = out_vec.shape[0]

    return seq_len, cd

def run(args):
    filepaths = sorted(glob.glob(os.path.join(args.src, "*.h5")))
    if args.num != -1:
        filepaths = filepaths[:args.num]

    save_path = args.src + '_pc_stat.txt'
    record_res = None
    if os.path.exists(save_path):
        response = input(save_path + ' already exists, overwrite? (y/n) ')
        if response == 'y':
            os.system("rm {}".format(save_path))
            record_res = None
        else:
            with open(save_path, 'r') as fp:
                record_res = fp.readlines()
                n_processed = len(record_res) - 3

    if args.parallel:
        # dists = Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x) for x in filepaths)
        dists = mp.Pool(8).map(process_one, tqdm(filepaths))
    elif args.cd_by_sequence_length:
        # ret = Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x, is_sl=True) for x in filepaths)

        # lens, dists = zip(*Parallel(n_jobs=8, verbose=2)(delayed(process_one_sl)(x) for x in filepaths))
        lens, dists = zip(*mp.Pool(8).map(process_one_sl, tqdm(filepaths)))
        cd_sl = defaultdict(list)
        for seq_len, cd in zip(lens, dists):
            if cd is not None:
                cd_sl[seq_len].append(cd)
        mean = lambda x: x  # seq_len: [sl, sl, sl, ...]
        cd_sl = {k: mean(v) for k, v in cd_sl.items()}
        dic_save_path = os.path.splitext(save_path)[0] + '_by_seq_len.pkl'
        with open(dic_save_path, 'wb') as f:
            pickle.dump(cd_sl, f)
    else:
        dists = []
        for i in range(len(filepaths)):
            print("processing[{}] {}".format(i, filepaths[i]))
            data_id = filepaths[i].split('/')[-1].split('.')[0]

            if record_res is not None and i < n_processed:
                record_dist = record_res[i].split('\t')[-1][:-1]
                record_dist = None if record_dist == 'None' else eval(record_dist)
                dists.append(record_dist)
                continue

            if data_id in SKIP_DATA:
                print("skip {}".format(data_id))
                res = None
            else:
                res = process_one(filepaths[i])
            with open(save_path, 'a') as fp:
                print("{}\t{}\t{}".format(i, data_id, res), file=fp)
            dists.append(res)
            

    valid_dists = [x for x in dists if x is not None]
    valid_dists = sorted(valid_dists)
    print("top 20 largest error:")
    print(valid_dists[-20:][::-1])
    n_valid = len(valid_dists)
    n_invalid = len(dists) - n_valid

    avg_dist = np.mean(valid_dists)
    trim_avg_dist = np.mean(valid_dists[int(n_valid * 0.1):-int(n_valid * 0.1)])
    med_dist = np.median(valid_dists)

    print("#####" * 10)
    print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths))
    print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist)
    with open(save_path, "a") as fp:
        print("#####" * 10, file=fp)
        print("total:", len(filepaths), "\t invalid:", n_invalid, "\t invalid ratio:", n_invalid / len(filepaths),
              file=fp)
        print("avg dist:", avg_dist, "trim_avg_dist:", trim_avg_dist, "med dist:", med_dist,
              file=fp)


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
parser.add_argument('--n_points', type=int, default=2000)
parser.add_argument('--num', type=int, default=-1)
parser.add_argument('--parallel', action='store_true', help="use parallelization")
parser.add_argument('--cd_by_sequence_length', action='store_true')
args = parser.parse_args()
assert not (args.parallel & args.cd_by_sequence_length)

print(args.src)
print("SKIP DATA:", SKIP_DATA)
since = time.time()
run(args)
end = time.time()
print("running time: {}s".format(end - since))