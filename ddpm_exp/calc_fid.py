from cleanfid import fid
import argparse
parser = argparse.ArgumentParser(description=globals()["__doc__"])
parser.add_argument('--path1', type=str, required=True, help='Path to the images')
parser.add_argument('--path2', type=str, required=True, help='Path to the images')
args = parser.parse_args()

if args.path2=="cifar10":
    score = fid.compute_fid(args.dir, dataset_name="cifar10", dataset_res=32, dataset_split="train")
else:
    score = fid.compute_fid(args.path1, args.path2)
print("FID: ", score)