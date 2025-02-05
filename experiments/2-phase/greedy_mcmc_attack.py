import copy
import math
import random
import sys
from collections import defaultdict
from itertools import count

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import autograd
from torch_geometric.utils import dense_to_sparse, from_networkx, to_networkx

sys.path.append("../../")

from src.attacks.greedy_gd import *
from src.models.gcn import *
from src.models.trainable import *
from src.utils.datasets import *

budget = None
ptb_rate = None


def gen_weights(size):
    lst = []
    for i in range(1, size + 1):
        lst.append(1 / (10 * math.log2(i + 1)))
    return lst


def constant_fn(delta, initial_loss, i, first_phase_edges):
    return abs(delta) < (initial_loss / 100)


def increasing_fn(delta, initial_loss, i, first_phase_edges):
    # print(abs(delta), (((i + first_phase_edges) * initial_loss) / (first_phase_edges * 100)), abs(delta) < (((i + 1) * initial_loss) / (first_phase_edges * 100)))
    return abs(delta) < (((i + 1) * initial_loss) / (first_phase_edges * 100))


def binary_fn(rand, i):
    return rand < 0.5


def decreasing_fn(rand, i):
    return rand < math.exp(-math.log(i + 1))


def manage_dirty_data(G_dirty, dirty_data_copy, device):
    dirty_data = from_networkx(G_dirty).to(device)
    dirty_data.x = dirty_data_copy.x
    dirty_data.y = dirty_data_copy.y
    dirty_data.train_mask = dirty_data_copy.train_mask
    dirty_data.test_mask = dirty_data_copy.test_mask

    return dirty_data


def manage_clean_data(G, data, device):
    modified_data = from_networkx(G).to(device)
    modified_data.x = data.x
    modified_data.y = data.y
    modified_data.train_mask = data.train_mask
    modified_data.test_mask = data.test_mask

    return modified_data


def manage_edge(G, u, v, hasEdge):
    if hasEdge:
        G.remove_edge(u, v)
    else:
        G.add_edge(u, v)


def attack(data, model, percent, device):
    attacker = Metattack(data, device=device)
    attacker.setup_surrogate(
        model,
        labeled_nodes=data.train_mask,
        unlabeled_nodes=data.test_mask,
        lambda_=0.0,
    )
    attacker.reset()
    attacker.attack(percent)

    return attacker


def two_phase_attack_mcmc(
    data,
    train,
    model,
    split,
    edges_to_add,
    rand_fn,
    accept_fn,
    device,
    seed=42,
    verbose=False,
):
    global budget, ptb_rate
    initial_loss, initial_accuracy = train.test(data)
    random.seed(seed)
    dirty_data_copy = copy.copy(data)
    diff_threshold = abs(initial_loss / 200)
    first_phase_edges = int(budget * split)
    second_phase_percent = ptb_rate * (1 - split) * 1 / 2
    accuracies, losses = [initial_accuracy], [initial_loss]
    G = to_networkx(data, to_undirected=True)
    G_dirty = to_networkx(data, to_undirected=True)

    # run a metattack on 1 - split
    # store those edges in a dirty matrix
    # calculate loss + accuracy of dirty matrix

    dirty_data = from_networkx(G_dirty).to(device)
    dirty_data.x = dirty_data_copy.x
    dirty_data.y = dirty_data_copy.y
    dirty_data.train_mask = dirty_data_copy.train_mask
    dirty_data.test_mask = dirty_data_copy.test_mask

    attacker_dirty = Metattack(dirty_data, device=device)
    attacker_dirty.setup_surrogate(
        model,
        labeled_nodes=dirty_data_copy.train_mask,
        unlabeled_nodes=dirty_data_copy.test_mask,
        lambda_=0.0,
    )
    attacker_dirty.reset()
    attacker_dirty.attack(second_phase_percent)

    degs_dirty = defaultdict(tuple)

    for k, v in attacker_dirty._added_edges.items():
        degs_dirty[v] = (k, True)

    for k, v in attacker_dirty._removed_edges.items():
        degs_dirty[v] = (k, False)

    for _, second in degs_dirty.items():
        u, v = second[0]
        if second[1]:
            G_dirty.add_edge(u, v)
        else:
            G_dirty.remove_edge(u, v)

    dirty_data = from_networkx(G_dirty).to(device)
    dirty_data.x = dirty_data_copy.x
    dirty_data.y = dirty_data_copy.y
    dirty_data.train_mask = dirty_data_copy.train_mask
    dirty_data.test_mask = dirty_data_copy.test_mask

    initial_dirty_loss, initial_dirty_accuracy = train.test(dirty_data)

    degs_set = set([v[0] for v in degs_dirty.values()])

    # remove dirty edges from edges_to_ad
    data_copy = copy.copy(data)
    i, j = 0, 0  # i - number added, j - attempts
    dirty_prev_loss, prev_loss = initial_dirty_loss, initial_loss
    weights = gen_weights(len(edges_to_add))
    while i < first_phase_edges:
        if verbose and i % 10 == 0:
            print(f"Attempt: {j}, Selected: {i}")
        j += 1
        u, v = random.choices(edges_to_add, weights=weights, k=1)[0]
        # u, v = random.choices(edges_to_add, k=1)[0]
        if (u, v) in degs_set:
            continue

        hasEdge = G.has_edge(u, v)
        # u, v = edges_to_add[j]

        # clean matrix
        if hasEdge:
            G.remove_edge(u, v)
        else:
            G.add_edge(u, v)

        modified_data = from_networkx(G).to(device)
        modified_data.x = data.x
        modified_data.y = data.y
        modified_data.train_mask = data.train_mask
        modified_data.test_mask = data.test_mask

        modified_loss, modified_accuracy = train.test(modified_data)
        delta = modified_loss - initial_loss

        # dirty matrix
        if hasEdge:
            G_dirty.remove_edge(u, v)
        else:
            G_dirty.add_edge(u, v)

        dirty_data = from_networkx(G_dirty).to(device)
        dirty_data.x = dirty_data_copy.x
        dirty_data.y = dirty_data_copy.y
        dirty_data.train_mask = dirty_data_copy.train_mask
        dirty_data.test_mask = dirty_data_copy.test_mask

        dirty_loss, dirty_accuracy = train.test(dirty_data)
        dirty_delta = dirty_loss - dirty_prev_loss
        master_dirty = dirty_loss - initial_dirty_loss

        # if abs(delta) > 1/200 loss, immediately continue
        if verbose and i % 10 == 0:
            print(
                f"max_change: {initial_loss / 100}, master_clean_delta: {delta}, master_dirty_delta: {master_dirty}"
            )
        # modified_loss: {modified_loss}, initial_loss: {initial_loss}, dirty_delta: {dirty_delta}")
        # consider something sublinear here
        if accept_fn(delta, initial_loss, i, first_phase_edges) and dirty_delta > 0:
            # print("works, adding")
            i += 1
            dirty_prev_loss = dirty_loss
            accuracies.append(modified_accuracy)
            losses.append(modified_loss)
        elif (
            not accept_fn(delta, initial_loss, i, first_phase_edges)
            or master_dirty < delta
            or master_dirty < 0
        ):
            if hasEdge:
                G.add_edge(u, v)
                G_dirty.add_edge(u, v)
            else:
                G.remove_edge(u, v)
                G_dirty.remove_edge(u, v)
            continue
        else:
            rnd = random.random()
            if rand_fn(rnd, i):
                # print("selected prob ltf")
                i += 1
                dirty_prev_loss = dirty_loss
                accuracies.append(modified_accuracy)
                losses.append(modified_loss)
            else:
                # print("removing edge")
                if hasEdge:
                    G.add_edge(u, v)
                    G_dirty.add_edge(u, v)
                else:
                    G.remove_edge(u, v)
                    G_dirty.remove_edge(u, v)

    modified_data = from_networkx(G).to(device)
    modified_data.x = data.x
    modified_data.y = data.y
    modified_data.train_mask = data.train_mask
    modified_data.test_mask = data.test_mask

    attacker = Metattack(modified_data, device=device)
    attacker.setup_surrogate(
        model,
        labeled_nodes=data.train_mask,
        unlabeled_nodes=data.test_mask,
        lambda_=0.0,
    )
    attacker.reset()
    attacker.attack(second_phase_percent)

    degs = defaultdict(tuple)

    for k, v in attacker._added_edges.items():
        degs[v] = (k, True)

    for k, v in attacker._removed_edges.items():
        degs[v] = (k, False)

    for _, second in degs.items():
        u, v = second[0]
        if second[1]:
            G.add_edge(u, v)
        else:
            G.remove_edge(u, v)

        modified_data = from_networkx(G).to(device)
        modified_data.x = data.x
        modified_data.y = data.y
        modified_data.train_mask = data.train_mask
        modified_data.test_mask = data.test_mask

        modified_loss, modified_accuracy = train.test(modified_data)

        # accuracies.append(modified_accuracy)
        accuracies.append(modified_accuracy)
        losses.append(modified_loss)

    # print(accuracies)
    return accuracies, losses, j / (first_phase_edges + 1)
    # accuracies, losses = [initial_accuracy], [initial_loss]
    # degs_dirty = defaultdict(tuple)
    # degs = defaultdict(tuple)
    # random.seed(seed)

    # diff_threshold = abs(initial_loss / 200)
    # first_phase_edges = int(budget * split)
    # second_phase_percent = ptb_rate * (1 - split) * 1 / 2

    # dirty_data_copy = copy.copy(data)
    # G = to_networkx(data, to_undirected=True)
    # G_dirty = to_networkx(data, to_undirected=True)

    # # run a metattack on 1 - split
    # # store those edges in a dirty matrix
    # # calculate loss + accuracy of dirty matrix

    # dirty_data = manage_dirty_data(G_dirty, dirty_data_copy, device)

    # if split > 0:
    #     attacker_dirty = attack(dirty_data, model, second_phase_percent, device)

    #     for k, v in attacker_dirty._added_edges.items():
    #         degs_dirty[v] = (k, True)

    #     for k, v in attacker_dirty._removed_edges.items():
    #         degs_dirty[v] = (k, False)

    #     for _, second in degs_dirty.items():
    #         u, v = second[0]
    #         if second[1]:
    #             G_dirty.add_edge(u, v)
    #         else:
    #             G_dirty.remove_edge(u, v)

    #     degs_set = set([v[0] for v in degs_dirty.values()])

    # manage_dirty_data(G_dirty, dirty_data_copy, device)
    # print(G.number_of_edges(), G_dirty.number_of_edges())
    # initial_dirty_loss, initial_dirty_accuracy = train.test(dirty_data)
    # print(initial_dirty_loss)
    # dirty_prev_loss, prev_loss = initial_dirty_loss, initial_loss

    # i, j = 0, 0  # i - number added, j - attempts
    # weights = gen_weights(len(edges_to_add))

    # while i < first_phase_edges:
    #     if verbose:
    #         print(f"Attempt: {j}, Selected: {i}, Edges: {G.number_of_edges()}")
    #     j += 1
    #     u, v = random.choices(edges_to_add, weights=weights, k=1)[0]

    #     if (u, v) in degs_set:
    #         continue

    #     matches = G.has_edge(u, v) == G_dirty.has_edge(u, v)
    #     if not matches:
    #         continue

    #     hasEdge = G.has_edge(u, v)

    #     # clean matrix
    #     if hasEdge:
    #         G.remove_edge(u, v)
    #     else:
    #         G.add_edge(u, v)

    #     modified_data = manage_clean_data(G, data, device)
    #     modified_loss, modified_accuracy = train.test(modified_data)

    #     delta = modified_loss - initial_loss

    #     # dirty matrix
    #     if hasEdge:
    #         G_dirty.remove_edge(u, v)
    #     else:
    #         G_dirty.add_edge(u, v)

    #     dirty_data = manage_dirty_data(G_dirty, dirty_data_copy, device)

    #     dirty_loss, dirty_accuracy = train.test(dirty_data)
    #     print(dirty_loss)
    #     dirty_delta = dirty_loss - dirty_prev_loss
    #     master_dirty = dirty_loss - initial_dirty_loss

    #     if verbose:
    #         print(
    #             f"max_change: {initial_loss / 100}, master_clean_delta: {delta}, master_dirty_delta: {master_dirty}"
    #         )

    #     if accept_fn(delta, initial_loss, i, first_phase_edges) and dirty_delta > 0:
    #         i += 1
    #         dirty_prev_loss = dirty_loss
    #         accuracies.append(modified_accuracy)
    #         losses.append(modified_loss)
    #     elif (
    #         not accept_fn(delta, initial_loss, i, first_phase_edges)
    #         or master_dirty < delta
    #         or master_dirty < 0
    #     ):
    #         if hasEdge:
    #             G.add_edge(u, v)
    #             G_dirty.add_edge(u, v)
    #         else:
    #             G.remove_edge(u, v)
    #             G_dirty.remove_edge(u, v)
    #         continue
    #     else:
    #         rnd = random.random()
    #         if rand_fn(rnd, i):
    #             # print("selected prob ltf")
    #             i += 1
    #             dirty_prev_loss = dirty_loss
    #             accuracies.append(modified_accuracy)
    #             losses.append(modified_loss)
    #         else:
    #             # print("removing edge")
    #             if hasEdge:
    #                 G.add_edge(u, v)
    #                 G_dirty.add_edge(u, v)
    #             else:
    #                 G.remove_edge(u, v)
    #                 G_dirty.remove_edge(u, v)

    # modified_data = manage_clean_data(G, data, device)

    # attacker = attack(modified_data, model, second_phase_percent, device)

    # for k, v in attacker._added_edges.items():
    #     degs[v] = (k, True)

    # for k, v in attacker._removed_edges.items():
    #     degs[v] = (k, False)

    # for _, second in degs.items():
    #     u, v = second[0]
    #     if second[1]:
    #         G.add_edge(u, v)
    #     else:
    #         G.remove_edge(u, v)

    #     modified_data = manage_clean_data(G, data, device)

    #     modified_loss, modified_accuracy = train.test(modified_data)

    #     accuracies.append(modified_accuracy)
    #     losses.append(modified_loss)

    # return accuracies, losses


def load_model_and_edges(
    model_save_path, list_save_path, shape, classes, layers, device
):
    loaded_model_state_dict = torch.load(model_save_path)

    model = GCN(shape, classes, layers).to(device)
    model.load_state_dict(loaded_model_state_dict)

    edges_to_add = torch.load(list_save_path)

    train = Trainable(model)

    return model, edges_to_add, train


def initialize(data, _ptb_rate=0.15):
    global ptb_rate, budget
    ptb_rate = _ptb_rate
    G = to_networkx(data, to_undirected=True)
    initial_edge_count = G.number_of_edges() // 2
    budget = int(ptb_rate * initial_edge_count)

    return G, initial_edge_count, ptb_rate, budget


def plot_results(dic, ptb_rate, title_one, title_two, title_three, t):
    plt.figure(figsize=(10, 6))

    # Iterate over the dictionary and plot each list
    for label, values in dic.items():
        plt.plot(values, label=str(label))

    plt.xlabel("# of edges inserted")
    plt.ylabel(t)
    plt.title(
        f"{t} vs # of edges inserted (ptb rate = {ptb_rate}), {title_one} method, {title_two} decision, {title_three} acceptance"
    )
    plt.legend()

    plt.show()


def collect_edges(model, data, device):
    # run 5 metattacks w/ ptb of 1
    amts = defaultdict(int)

    for _ in range(10):
        attacker = Metattack(data, device=device)
        attacker.setup_surrogate(
            model,
            labeled_nodes=data.train_mask,
            unlabeled_nodes=data.test_mask,
            lambda_=0.0,
        )
        attacker.reset()
        attacker.attack(0.1)

        for edge in attacker._added_edges.keys():
            amts[edge] += 1

    sorted_list = sorted(amts.items(), key=lambda item: item[1], reverse=True)
    sorted_keys = [key for key, value in sorted_list]

    return sorted_keys


# def two_phase_attack(split):
def two_phase_attack_greedy(
    data,
    train,
    model,
    split,
    edges_to_add,
    accept_fn,
    device,
    seed=42,
    verbose=False,
):
    global budget, ptb_rate
    initial_loss, initial_accuracy = train.test(data)
    dirty_data_copy = copy.copy(data)
    random.seed(seed)
    diff_threshold = abs(initial_loss / 200)
    first_phase_edges = int(budget * split)
    second_phase_percent = ptb_rate * (1 - split) * 1 / 2
    accuracies, losses = [initial_accuracy], [initial_loss]
    G = to_networkx(data, to_undirected=True)
    G_dirty = to_networkx(data, to_undirected=True)

    dirty_data = from_networkx(G_dirty).to(device)
    dirty_data.x = dirty_data_copy.x
    dirty_data.y = dirty_data_copy.y
    dirty_data.train_mask = dirty_data_copy.train_mask
    dirty_data.test_mask = dirty_data_copy.test_mask

    attacker_dirty = Metattack(dirty_data, device=device)
    attacker_dirty.setup_surrogate(
        model,
        labeled_nodes=dirty_data_copy.train_mask,
        unlabeled_nodes=dirty_data_copy.test_mask,
        lambda_=0.0,
    )
    attacker_dirty.reset()
    attacker_dirty.attack(second_phase_percent)

    degs_dirty = defaultdict(tuple)

    for k, v in attacker_dirty._added_edges.items():
        degs_dirty[v] = (k, True)

    for k, v in attacker_dirty._removed_edges.items():
        degs_dirty[v] = (k, False)

    for _, second in degs_dirty.items():
        u, v = second[0]
        if second[1]:
            G_dirty.add_edge(u, v)
        else:
            G_dirty.remove_edge(u, v)

    dirty_data = from_networkx(G_dirty).to(device)
    dirty_data.x = dirty_data_copy.x
    dirty_data.y = dirty_data_copy.y
    dirty_data.train_mask = dirty_data_copy.train_mask
    dirty_data.test_mask = dirty_data_copy.test_mask

    initial_dirty_loss, initial_dirty_accuracy = train.test(dirty_data)

    degs_set = set([v[0] for v in degs_dirty.values()])

    weights = gen_weights(len(edges_to_add))
    data_copy = copy.copy(data)
    i, j = 0, 0  # i - number added, j - spot in list
    dirty_prev_loss, prev_loss = initial_dirty_loss, initial_loss
    while i < first_phase_edges:
        if verbose and i % 10 == 0:
            print(f"Attempt: {j}, Selected: {i}")
        u, v = random.choices(edges_to_add, weights=weights, k=1)[0]

        matches = G.has_edge(u, v) == G_dirty.has_edge(u, v)
        if not matches:
            continue
            
        hasEdge = G.has_edge(u, v)

        if (u, v) in degs_set:
            continue

        # clean
        if hasEdge:
            G.remove_edge(u, v)
        else:
            G.add_edge(u, v)

        modified_data = from_networkx(G).to(device)
        modified_data.x = data.x
        modified_data.y = data.y
        modified_data.train_mask = data.train_mask
        modified_data.test_mask = data.test_mask

        modified_loss, modified_accuracy = train.test(modified_data)
        delta = modified_loss - initial_loss

        # dirty matrix
        if hasEdge:
            G_dirty.remove_edge(u, v)
        else:
            G_dirty.add_edge(u, v)

        dirty_data = from_networkx(G_dirty).to(device)
        dirty_data.x = dirty_data_copy.x
        dirty_data.y = dirty_data_copy.y
        dirty_data.train_mask = dirty_data_copy.train_mask
        dirty_data.test_mask = dirty_data_copy.test_mask

        dirty_loss, dirty_accuracy = train.test(dirty_data)
        dirty_delta = dirty_loss - dirty_prev_loss
        master_dirty = dirty_loss - initial_dirty_loss
        # print(modified_loss)

        # if (abs(modified_loss - initial_loss) / max(modified_loss, initial_loss)) <= diff_threshold:
        # def constant_fn(delta, initial_loss, i, first_phase_edges):
        if (
            accept_fn(abs(delta), initial_loss, i, first_phase_edges)
            and dirty_delta > 0
        ):
            # if modified_accuracy == initial_accuracy:
            # print(modified_accuracy, i)
            i += 1
            # accuracies.append(modified_accuracy)
            losses.append(modified_loss)
            accuracies.append(modified_accuracy)
        else:
            # print(i, 'miss!')
            if hasEdge:
                G.add_edge(u, v)
                G_dirty.add_edge(u, v)
            else:
                G.remove_edge(u, v)
                G_dirty.remove_edge(u, v)

        j += 1

    modified_data = from_networkx(G).to(device)
    modified_data.x = data.x
    modified_data.y = data.y
    modified_data.train_mask = data.train_mask
    modified_data.test_mask = data.test_mask

    attacker = Metattack(modified_data, device=device)
    attacker.setup_surrogate(
        model,
        labeled_nodes=data.train_mask,
        unlabeled_nodes=data.test_mask,
        lambda_=0.0,
    )
    attacker.reset()
    attacker.attack(second_phase_percent)

    degs = defaultdict(tuple)

    for k, v in attacker._added_edges.items():
        degs[v] = (k, True)

    for k, v in attacker._removed_edges.items():
        degs[v] = (k, False)

    for _, second in degs.items():
        u, v = second[0]
        if second[1]:
            G.add_edge(u, v)
        else:
            G.remove_edge(u, v)

        modified_data = from_networkx(G).to(device)
        modified_data.x = data.x
        modified_data.y = data.y
        modified_data.train_mask = data.train_mask
        modified_data.test_mask = data.test_mask

        modified_loss, modified_accuracy = train.test(modified_data)

        # accuracies.append(modified_accuracy)
        losses.append(modified_loss)
        accuracies.append(modified_accuracy)

    return accuracies, losses, j / (first_phase_edges + 1)