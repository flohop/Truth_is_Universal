import torch as t
import pandas as pd
import numpy as np
import os
from glob import glob
import random

ROOT = os.path.dirname(os.path.abspath(__file__))
ACTS_BATCH_SIZE = 25

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
        base_name = dataset_name.replace('_conj', '').replace('_disj', '').replace('neg_', '')
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

     




