from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

import torch
import torch as t
import numpy as np

from utils import dataset_sizes, collect_training_data
import ray
from ray import tune

# -----------------------------------------------------------
# Feature Interaction Helper
# -----------------------------------------------------------
def make_interaction_features(acts_2d, interaction_names):
    """Generate interaction features dynamically from names."""
    feature_dict = {
        'proj_t_g': acts_2d[:, 0:1],
        'proj_t_p': acts_2d[:, 1:2],
        'proj_p':   acts_2d[:, 2:3],
        'inter1':   acts_2d[:, 3:4]
    }

    features = []
    for name in interaction_names:
        if '+' in name:
            a, b = name.split('+')
            features.append(feature_dict[a] + feature_dict[b])
        elif '-' in name:
            a, b = name.split('-')
            features.append(feature_dict[a] - feature_dict[b])
        elif '*' in name:
            a, b = name.split('*')
            features.append(feature_dict[a] * feature_dict[b])
        else:
            features.append(feature_dict[name])
    return np.concatenate(features, axis=1)

# -----------------------------------------------------------
# TTPDTest Probe
# -----------------------------------------------------------
class TTPDTest:
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    def _project_acts(self, acts):
        proj_t_g = (acts.numpy() @ self.t_g.numpy())[:, None]
        proj_p = acts.numpy() @ self.polarity_direc.T
        proj_t_p = (acts.numpy() @ self.t_p.numpy())[:, None]
        inter1 = proj_t_p * proj_p
        return np.concatenate([proj_t_g, proj_t_p, proj_p, inter1], axis=1)

# -----------------------------------------------------------
# Direction Learners
# -----------------------------------------------------------
def learn_truth_directions(acts_centered, labels, polarities):
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    labels_copy = t.where(labels == 0, t.tensor(-1.0), labels)
    X = labels_copy.reshape(-1,1) if all_polarities_zero else t.column_stack([labels_copy, labels_copy*polarities])
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered
    if all_polarities_zero:
        return solution.flatten(), None
    else:
        return solution[0,:], solution[1,:]

def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1] = 0
    LR = LogisticRegression(penalty='l2', C=0.1, fit_intercept=True, solver='lbfgs', max_iter=2000)
    LR.fit(acts.numpy(), polarities_copy.numpy())
    return LR.coef_

# -----------------------------------------------------------
# Training function for Ray Tune
# -----------------------------------------------------------
def train_ttpd_test(config, acts_centered_train, acts_train, labels_train, polarities_train,
                    acts_centered_val, acts_val, labels_val, polarities_val):
    probe = TTPDTest()
    probe.t_g, probe.t_p = learn_truth_directions(acts_centered_train, labels_train, polarities_train)
    probe.polarity_direc = learn_polarity_direction(acts_train, polarities_train)

    # Project and compute interaction features
    acts_2d_train = probe._project_acts(acts_train)
    acts_2d_val = probe._project_acts(acts_val)

    X_train = make_interaction_features(acts_2d_train, config['interaction_features'])
    X_val   = make_interaction_features(acts_2d_val, config['interaction_features'])

    # Build pipeline
    steps = []
    if config['use_scaler']:
        steps.append(('scaler', StandardScaler()))
    steps.append(('poly', PolynomialFeatures(degree=config['degree'], include_bias=config['include_bias'])))
    steps.append(('lr', LogisticRegression(
        penalty='l2',
        C=config['C'],
        fit_intercept=config['fit_intercept'],
        solver=config['solver'],
        max_iter=2000
    )))
    probe.LR = Pipeline(steps)

    # Fit and evaluate
    probe.LR.fit(X_train, labels_train.numpy())
    preds_val = probe.LR.predict(X_val)
    acc_val = (preds_val == labels_val.numpy()).mean()

    tune.report({"accuracy": acc_val})

# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == '__main__':
    model_family = 'Llama3'
    model_size = '8B'
    model_type = 'chat'
    layer = 12
    device = 'mps' if torch.has_mps else 'cpu'

    train_sets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors",
                  "animal_class", "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]
    val_sets = ["cities_conj", "cities_disj", "sp_en_trans_conj", "sp_en_trans_disj",
                "inventors_conj", "inventors_disj", "animal_class_conj", "animal_class_disj",
                "element_symb_conj", "element_symb_disj", "facts_conj", "facts_disj",
                "common_claim_true_false", "counterfact_true_false"]

    train_set_sizes = dataset_sizes(train_sets)
    val_set_sizes = dataset_sizes(val_sets)

    acts_centered_train, acts_train, labels_train, polarities_train = collect_training_data(
        train_sets, train_set_sizes, model_family, model_size, model_type, layer
    )
    acts_centered_val, acts_val, labels_val, polarities_val = collect_training_data(
        val_sets, val_set_sizes, model_family, model_size, model_type, layer
    )

    ray.init(ignore_reinit_error=True)

    search_space = {
        'C': tune.loguniform(1e-3, 10),
        'fit_intercept': tune.choice([True, False]),
        'solver': tune.choice(['lbfgs', 'saga', 'liblinear']),
        'degree': tune.choice([1,2,3]),
        'include_bias': tune.choice([True, False]),
        'use_scaler': tune.choice([True, False]),
        'interaction_features': tune.choice([
            ['proj_t_g', 'proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p'],
            ['proj_t_g', 'proj_t_p', 'inter1'],
            ['proj_t_g', 'proj_t_p', 'proj_t_g+proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_t_g-proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_t_g*proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_g*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_p*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_g*proj_t_p*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1'],
        ])
    }

    analysis = tune.run(
        tune.with_parameters(
            train_ttpd_test,
            acts_centered_train=acts_centered_train,
            acts_train=acts_train,
            labels_train=labels_train,
            polarities_train=polarities_train,
            acts_centered_val=acts_centered_val,
            acts_val=acts_val,
            labels_val=labels_val,
            polarities_val=polarities_val
        ),
        resources_per_trial={"cpu": 2},
        metric="accuracy",
        mode="max",
        num_samples=20,
        config=search_space
    )

    print("Best config found:", analysis.best_config)
