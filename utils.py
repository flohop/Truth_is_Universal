import configparser

import torch as t
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from glob import glob
import random
import einops


from matplotlib import pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# --- Basic Config and Constants ---
config = configparser.ConfigParser()
# Assumes config.ini is in the same directory
try:
    config.read('config.ini')
except:
    print("Warning: config.ini not found. Model paths may not load correctly.")


# --- Model Loading ---
def load_model(model_family: str, model_size: str, model_type: str, device: str):
    """Loads a model and tokenizer from the specified family, size, and type."""
    model_path = os.path.join(config[model_family]['weights_directory'],
                              config[model_family][f'{model_size}_{model_type}_subdir'])

    if model_family == 'Llama2':
        tokenizer = LlamaTokenizer.from_pretrained(str(model_path))
        model = LlamaForCausalLM.from_pretrained(str(model_path))
        tokenizer.bos_token = '<s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(str(model_path))

    if model_family == "Gemma2":
        model = model.to(t.bfloat16)
    else:
        model = model.half()

    return tokenizer, model.to(device)

def plot_lr_feature_importance(coefs, feature_names=None, title="Feature Importance (|Coefficient|)"):
    """
    Plot absolute logistic regression coefficients as feature importance.

    Parameters
    ----------
    lr_model : sklearn.linear_model.LogisticRegression
        A trained logistic regression model.
    feature_names : list of str, optional
        Names of the features corresponding to the coefficients.
    title : str
        Title for the plot.
    """
    n_features = len(coefs)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    plt.figure(figsize=(5, 3))
    plt.bar(feature_names, coefs, color="skyblue", edgecolor="black")
    plt.ylabel("Absolute Coefficient Magnitude")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

def collect_acts(dataset_name, model_family, model_size,
                  model_type, layer, center=True, scale=False, device='cpu'):
    """
    Collects activations from a dataset of statements, returns as a tensor of shape [n_activations, activation_dimension].
    """
    directory = os.path.join(ROOT, 'acts', model_family, model_size, model_type, dataset_name)
    activation_files = glob(os.path.join(directory, f'layer_{layer}_*.pt'))
    acts = [t.load(os.path.join(directory, f'layer_{layer}_{i}.pt'), map_location=device) for i in range(0, ACTS_BATCH_SIZE * len(activation_files), ACTS_BATCH_SIZE)] 
    try:
        acts = t.cat(acts, dim=0).to(device)
    except:
        raise Exception("No activation vectors could be found for the dataset " 
                        + dataset_name + ". Please generate them first using generate_acts.")
    if center:
        acts = acts - t.mean(acts, dim=0)
    if scale:
        acts = acts / t.std(acts, dim=0)
    return acts

def cat_data(d):
    """
    Given a dict of datasets (possible recursively nested), returns the concatenated activations and labels.
    """
    all_acts, all_labels = [], []
    for dataset in d:
        if isinstance(d[dataset], dict):
            if len(d[dataset]) != 0: # disregard empty dicts
                acts, labels = cat_data(d[dataset])
                all_acts.append(acts), all_labels.append(labels)
        else:
            acts, labels = d[dataset]
            all_acts.append(acts), all_labels.append(labels)
    try:
        acts, labels = t.cat(all_acts, dim=0), t.cat(all_labels, dim=0)
    except:
        raise Exception("No activation vectors could be found for this dataset. Please generate them first using generate_acts.")
    return acts, labels

class DataManager:
    """
    Class for storing activations and labels from datasets of statements.
    """
    def __init__(self):
        self.data = {
            'train' : {},
            'val' : {}
        } # dictionary of datasets
        self.proj = None # projection matrix for dimensionality reduction
    
    def add_dataset(self, dataset_name, model_family, model_size, model_type, layer,
                     label='label', split=None, seed=None, center=True, scale=False, device='cpu'):
        """
        Add a dataset to the DataManager.
        label : which column of the csv file to use as the labels.
        If split is not None, gives the train/val split proportion. Uses seed for reproducibility.
        """
        acts = collect_acts(dataset_name, model_family, model_size, model_type,
                             layer, center=center, scale=scale, device=device)
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))
        labels = t.Tensor(df[label].values).to(device)

        if split is None:
            self.data[dataset_name] = acts, labels

        if split is not None:
            assert 0 <= split and split <= 1
            if seed is None:
                seed = random.randint(0, 1000)
            t.manual_seed(seed)
            train = t.randperm(len(df)) < int(split * len(df))
            val = ~train
            self.data['train'][dataset_name] = acts[train], labels[train]
            self.data['val'][dataset_name] = acts[val], labels[val]

    def get(self, datasets):
        """
        Output the concatenated activations and labels for the specified datasets.
        datasets : can be 'all', 'train', 'val', a list of dataset names, or a single dataset name.
        If proj, projects the activations using the projection matrix.
        """
        if datasets == 'all':
            data_dict = self.data
        elif datasets == 'train':
            data_dict = self.data['train']
        elif datasets == 'val':
            data_dict = self.data['val']
        elif isinstance(datasets, list):
            data_dict = {}
            for dataset in datasets:
                if dataset[-6:] == ".train":
                    data_dict[dataset] = self.data['train'][dataset[:-6]]
                elif dataset[-4:] == ".val":
                    data_dict[dataset] = self.data['val'][dataset[:-4]]
                else:
                    data_dict[dataset] = self.data[dataset]
        elif isinstance(datasets, str):
            data_dict = {datasets : self.data[datasets]}
        else:
            raise ValueError(f"datasets must be 'all', 'train', 'val', a list of dataset names, or a single dataset name, not {datasets}")
        acts, labels = cat_data(data_dict)
        # if proj and self.proj is not None:
        #     acts = t.mm(acts, self.proj)
        return acts, labels


def dataset_sizes(dataset_names):
    """Modified to use absolute paths for Ray compatibility"""
    # Get the absolute path to your project directory
    # Adjust this to point to your project root
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # If in a .py file
    # OR for notebooks:
    # PROJECT_ROOT = os.getcwd()

    sizes = {}
    for dataset_name in dataset_names:
        # base_name = dataset_name.replace('_conj', '').replace('_disj', '').replace('neg_', '')
        base_name = dataset_name.replace('neg_', '')
        file_path = os.path.join(PROJECT_ROOT, 'datasets', f'{base_name}.csv')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        with open(file_path, 'r') as file:
            sizes[dataset_name] = sum(1 for _ in file) - 1  # Subtract header

    return sizes

def collect_training_data(dataset_names, train_set_sizes, model_family, model_size
                          , model_type, layer, **kwargs):
    """
    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities
    """
    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []
    
    for dataset_name in dataset_names:
        dm = DataManager()
        dm.add_dataset(dataset_name, model_family, model_size, model_type, layer, split=None, center=False, device='cpu')
        acts, labels = dm.data[dataset_name]
        
        polarity = -1.0 if 'neg_' in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)

        # balance the training dataset by including an equal number of activations from each dataset
        # choose the same subset of statements for affirmative and negated version of the dataset
        if 'neg_' not in dataset_name:
            subset_size = min(acts.shape[0], min(train_set_sizes.values()))
            rand_subset = np.random.choice(acts.shape[0], subset_size, replace=False)
        
        all_acts_centered.append(acts[rand_subset, :] - t.mean(acts[rand_subset, :], dim=0))
        all_acts.append(acts[rand_subset, :])
        all_labels.append(labels[rand_subset])
        all_polarities.append(polarities[rand_subset])

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))


def collect_training_data_tuner(dataset_names, train_set_sizes, model_family,
                                model_size, model_type, layer, seed=None, **kwargs):
    """
    Fixed version of collect_training_data with proper handling of paired datasets

    Takes as input the names of datasets in the format
    [affirmative_dataset1, negated_dataset1, affirmative_dataset2, negated_dataset2, ...]
    and returns a balanced training dataset of centered activations, activations, labels and polarities

    Args:
        seed: Random seed for reproducible subset selection
    """
    if seed is not None:
        np.random.seed(seed)

    all_acts_centered, all_acts, all_labels, all_polarities = [], [], [], []

    # Pre-generate random subsets for each affirmative dataset
    # This ensures the same subset is used for both affirmative and negated versions
    rand_subsets = {}
    for dataset_name in dataset_names:
        if 'neg_' not in dataset_name:
            # Load to get size
            dm = DataManager()
            dm.add_dataset(dataset_name, model_family, model_size, model_type, layer,
                           split=None, center=False, device='cpu')
            acts, _ = dm.data[dataset_name]

            # Generate random subset
            rand_subsets[dataset_name] = np.random.choice(
                acts.shape[0],
                min(train_set_sizes.values()),
                replace=False
            )

    # Now process all datasets with pre-computed subsets
    for dataset_name in dataset_names:
        dm = DataManager()
        dm.add_dataset(dataset_name, model_family, model_size, model_type, layer,
                       split=None, center=False, device='cpu')
        acts, labels = dm.data[dataset_name]

        polarity = -1.0 if 'neg_' in dataset_name else 1.0
        polarities = t.full((labels.shape[0],), polarity)

        # Get the appropriate subset
        if 'neg_' in dataset_name:
            # Use the subset from the corresponding affirmative dataset
            base_name = dataset_name.replace('neg_', '')
            rand_subset = rand_subsets[base_name]
        else:
            rand_subset = rand_subsets[dataset_name]

        all_acts_centered.append(acts[rand_subset, :] - t.mean(acts[rand_subset, :], dim=0))
        all_acts.append(acts[rand_subset, :])
        all_labels.append(labels[rand_subset])
        all_polarities.append(polarities[rand_subset])

    return map(t.cat, (all_acts_centered, all_acts, all_labels, all_polarities))


def compute_statistics(results):
    stats = {}
    for key in results:
        means = {dataset: np.mean(values) for dataset, values in results[key].items()}
        stds = {dataset: np.std(values) for dataset, values in results[key].items()}
        stats[key] = {'mean': means, 'std': stds}
    return stats

def compute_average_accuracies(results, num_iter):
    probe_stats = {}

    for probe_type in results:
        overall_means = []
        
        for i in range(num_iter):
            # Calculate mean accuracy for each dataset in this iteration
            iteration_means = [results[probe_type][dataset][i] for dataset in results[probe_type]]
            overall_means.append(np.mean(iteration_means))
        
        overall_means = np.array(overall_means)
        final_mean = np.mean(overall_means)
        std_dev = np.std(overall_means)
        
        probe_stats[probe_type.__name__] = {
            'mean': final_mean,
            'std_dev': std_dev
        }
    
    return probe_stats

## Utils regarding the toxicity paper
# Partly copied, partly adapted from: https://github.com/ajyl/dpo_toxic

def convert(orig_state_dict, cfg):
    state_dict = {}

    state_dict["embed.W_E"] = orig_state_dict["transformer.wte.weight"]
    state_dict["pos_embed.W_pos"] = orig_state_dict["transformer.wpe.weight"]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.weight"
        ]
        state_dict[f"blocks.{l}.ln1.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_1.bias"
        ]

        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = orig_state_dict[f"transformer.h.{l}.attn.c_attn.weight"]
        W_Q, W_K, W_V = t.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = orig_state_dict[f"transformer.h.{l}.attn.c_attn.bias"]
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = orig_state_dict[f"transformer.h.{l}.attn.c_proj.weight"]
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = orig_state_dict[
            f"transformer.h.{l}.attn.c_proj.bias"
        ]

        state_dict[f"blocks.{l}.ln2.w"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.weight"
        ]
        state_dict[f"blocks.{l}.ln2.b"] = orig_state_dict[
            f"transformer.h.{l}.ln_2.bias"
        ]

        W_in = orig_state_dict[f"transformer.h.{l}.mlp.c_fc.weight"]
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_fc.bias"
        ]

        W_out = orig_state_dict[f"transformer.h.{l}.mlp.c_proj.weight"]
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = orig_state_dict[
            f"transformer.h.{l}.mlp.c_proj.bias"
        ]
    state_dict["unembed.W_U"] = orig_state_dict["lm_head.weight"].T

    state_dict["ln_final.w"] = orig_state_dict["transformer.ln_f.weight"]
    state_dict["ln_final.b"] = orig_state_dict["transformer.ln_f.bias"]
    return state_dict

def get_svd(_model, toxic_vector, num_mlp_vecs):
    scores = []
    for layer in range(_model.cfg.n_layers):
        mlp_outs = _model.blocks[layer].mlp.W_out
        cos_sims = F.cosine_similarity(
            mlp_outs, toxic_vector.unsqueeze(0), dim=1
        )
        _topk = cos_sims.topk(k=300)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_vecs = [
        _model.blocks[x[2]].mlp.W_out[x[1]]
        for x in sorted_scores[:num_mlp_vecs]
    ]
    top_vecs = [x / x.norm() for x in top_vecs]
    _top_vecs = t.stack(top_vecs)

    svd = t.linalg.svd(_top_vecs.transpose(0, 1))
    return svd, sorted_scores

def load_hooked(model_name, weights_path):
    _model = HookedTransformer.from_pretrained(model_name,
                                               device="cpu",
                                               )
    cfg = _model.cfg


    _weights = t.load(weights_path, map_location=t.device("cpu"))[
        "state"
    ]
    weights = convert(_weights, cfg)
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(weights)
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model

     




