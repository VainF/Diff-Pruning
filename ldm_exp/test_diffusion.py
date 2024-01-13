import torch
import torch_fidelity

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input1', type=str)
parser.add_argument('--input2', type=str)

args = parser.parse_args()

metrics_dict = torch_fidelity.calculate_metrics(
    input1=args.input1, 
    input2=args.input2, 
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=True, 
    prc=True, 
    verbose=False,
    samples_find_deep=True
)
print(metrics_dict)