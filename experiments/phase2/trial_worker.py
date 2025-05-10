import torch
import argparse
import pickle

from greedy_mcmc_attack import *

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=float)
parser.add_argument("--trial", type=int)
parser.add_argument("--ptb", type=float)
parser.add_argument("--method", type=str)
parser.add_argument("--constant", type=bool)
parser.add_argument("--increasing", type=bool)
parser.add_argument("--binary", type=bool)
parser.add_argument("--decreasing", type=bool)
parser.add_argument("--dataset", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--list_path", type=str)
args = parser.parse_args()

s = args.split
j = args.trial
ptb = args.ptb
method = args.method
is_constant = args.constant
is_increasing = args.increasing
is_binary = args.binary
is_decreasing = args.decreasing
dataset = args.dataset
model_path = args.model_path
list_path = args.list_path

device = torch.device("cuda")

cora_dataset = Planetoid(root='', name=dataset)
data = cora_dataset[0].to(device)
model = GCN(data.x.shape[1], cora_dataset.num_classes, [16]).to(device)

model, edges_to_add, train = load_model_and_edges(model_path, list_path, model, device)

G, initial_edge_count, ptb_rate, budget = initialize(data, _ptb_rate=ptb)

# actually run...

acc, loss, itrs = two_phase_attack_greedy(
    data, train, model, s, edges_to_add, constant_fn, device=device,
    is_reversed=False, verbose=True
)
result = (j, acc, loss, itrs)

print(pickle.dumps(result).hex())