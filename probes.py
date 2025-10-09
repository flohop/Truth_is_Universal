from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import torch
import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sympy.physics.units import degree

from utils import dataset_sizes, collect_training_data

import ray
from ray import tune
from ray.tune.tune import BasicVariantGenerator


def train_ttpd_test(config, acts_centered, acts, labels, polarities):
    from copy import deepcopy

    # Split into train/validation
    acts_train, acts_val, labels_train, labels_val = train_test_split(
        acts, labels.numpy(), test_size=0.2, random_state=42
    )

    probe = TTPDTest()

    # Learn truth directions
    probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
    probe.t_g = probe.t_g.numpy()
    probe.t_p = probe.t_p.numpy()

    # Learn polarity direction
    probe.polarity_direc = learn_polarity_direction(acts, polarities)

    # Project activations
    acts_2d_train = probe._project_acts(acts_train)
    acts_2d_val = probe._project_acts(acts_val)

    # Build pipeline with hyperparameters from config
    probe.LR = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=config['degree'], include_bias=True)),
        ('lr', LogisticRegression(
            penalty='l2',
            C=config['C'],
            fit_intercept=config['fit_intercept'],
            solver=config['solver'],
            max_iter=2000
        ))
    ])

    # Fit and evaluate
    probe.LR.fit(acts_2d_train, labels_train)
    preds = probe.LR.predict(acts_2d_val)
    acc = accuracy_score(labels_val, preds)

    # Report to Ray Tune
    tune.report({"accuracy": acc})


def learn_truth_directions(acts_centered, labels, polarities):
    # Check if all polarities are zero (handling both int and float) -> if yes learn only t_g
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    # Make the sure the labels only have the values -1.0 and 1.0, replace 0 with -1
    labels_copy = labels.clone()
    labels_copy = t.where(labels_copy == 0.0, t.tensor(-1.0), labels_copy)
    
    if all_polarities_zero:
        X = labels_copy.reshape(-1, 1)
    else:
        X = t.column_stack([labels_copy, labels_copy * polarities])

    # Compute the analytical OLS solution
    # find the linear coefficients that best explain activations in terms of truth and polarity
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered

    # Extract t_g and t_p
    if all_polarities_zero:
        t_g = solution.flatten()
        t_p = None
    else:
        t_g = solution[0, :]
        t_p = solution[1, :]

    # weights that can be applied to the activation vector
    return t_g, t_p


def polarity_direction_lr(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty='l2', C=0.1, fit_intercept=True, solver='lbfgs', max_iter=2000)
    #LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    return LR_polarity

def learn_polarity_direction(acts, polarities):
    polarity_direc = polarity_direction_lr(acts, polarities).coef_
    return polarity_direc


def measure_polarity_direction_lr(train_acts, train_polarities, valid_acts, valid_polarities):
    LR = polarity_direction_lr(train_acts, train_polarities)

    pred = LR.predict(valid_acts)
    num_diff = np.sum(~np.isclose(pred, valid_polarities))
    return 1 -  num_diff / len(valid_acts)


class TTPDTest:
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPDTest()

        # Learn truth directions
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy()

        # Learn polarity direction
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

        # Project activations
        acts_2d = probe._project_acts(acts)

        # Full pipeline with scaling + polynomial features
        probe.LR = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=1, include_bias=True)),
            ('lr', LogisticRegression(C=0.1959, penalty='l2', fit_intercept=True, solver="saga", max_iter=2000))
        ])

        # Fit on all 4 features
        probe.LR.fit(acts_2d, labels.numpy())
        return probe

    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))

    def _project_acts(self, acts):
        proj_t_g = (acts.numpy() @ self.t_g)[:, None]  # general truth
        proj_p = (acts.numpy() @ self.polarity_direc.T)  # polarity
        proj_t_p = (acts.numpy() @ self.t_p)[:, None] * proj_p  # interaction
        # inter1 = proj_t_g + proj_t_p  # simple additive interaction
        acts_2d = np.concatenate((proj_t_g, proj_t_p, proj_p), axis=1)
        return acts_2d



class TTPD():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        # Learn direction for truth
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()

        # Learn direction for polarity
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

        # Project all dimensions onto the 2D truth/polarity space
        acts_2d = probe._project_acts(acts)

        # Fit logistic regression
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts_2d, labels.numpy())
        return probe

    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))

    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d

def ccs_loss(probe, acts, neg_acts):
    p_pos = probe(acts)
    p_neg = probe(neg_acts)
    consistency_losses = (p_pos - (1 - p_neg)) ** 2
    confidence_losses = t.min(t.stack((p_pos, p_neg), dim=-1), dim=-1).values ** 2
    return t.mean(consistency_losses + confidence_losses)


class CCSProbe(t.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=True),
            t.nn.Sigmoid()
        )
    
    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)
    
    def pred(self, acts, iid=None):
        return self(acts).round()
    
    def from_data(acts, neg_acts, labels=None, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, neg_acts = acts.to(device), neg_acts.to(device)
        probe = CCSProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ccs_loss(probe, acts, neg_acts)
            loss.backward()
            opt.step()

        if labels is not None: # flip direction if needed
            labels = labels.to(device)
            acc = (probe.pred(acts) == labels).float().mean()
            if acc < 0.5:
                probe.net[0].weight.data *= -1
        
        return probe

    @property
    def direction(self):
        return self.net[0].weight.data[0]
    
    @property
    def bias(self):
        return self.net[0].bias.data[0]
    

class LRProbe():
    def __init__(self):
        self.LR = None

    def from_data(acts, labels):
        probe = LRProbe()
        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)
        probe.LR.fit(acts.numpy(), labels.numpy())
        return probe

    def pred(self, acts):
        return t.tensor(self.LR.predict(acts))
    

class MMProbe(t.nn.Module):
    def __init__(self, direction, LR):
        super().__init__()
        self.direction = direction
        self.LR = LR

    def forward(self, acts):
        proj = acts @ self.direction
        return t.tensor(self.LR.predict(proj[:, None]))

    def pred(self, x):
        return self(x).round()

    def from_data(acts, labels, device='cpu'):
        acts, labels
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None], labels)
        
        probe = MMProbe(direction, LR).to(device)

        return probe

# (title, object)
TTPD_TYPES = [
        # ("TTPD", TTPD),
        ("TTPDAffirm", TTPDTest)
            ]

# To speed up testing, for full report use all probes
# ALL_PROBES = TTPD_TYPES + [("LRProbe", LRProbe), ("CCSProbe", CCSProbe), ("MMProbe", MMProbe)]
ALL_PROBES = TTPD_TYPES


if __name__ == '__main__':
    model_family = 'Llama3'  # options are 'Llama3', 'Llama2', 'Gemma', 'Gemma2' or 'Mistral'
    model_size = '8B'
    model_type = 'chat'  # options are 'chat' or 'base'
    layer = 12  # layer from which to extract activations

    device = 'mps' if torch.mps.is_available() else 'cpu'  # gpu speeds up CCS training a fair bit but is not required

    # define datasets used for training
    train_sets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors",
                  "animal_class",
                  "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]
    train_set_sizes = dataset_sizes(train_sets)

    val_sets = ["cities_conj", "cities_disj", "sp_en_trans_conj", "sp_en_trans_disj",
                "inventors_conj", "inventors_disj", "animal_class_conj", "animal_class_disj",
                "element_symb_conj", "element_symb_disj", "facts_conj", "facts_disj",
                "common_claim_true_false", "counterfact_true_false"]

    # load training data
    acts_centered, acts, labels, polarities = collect_training_data(train_sets, train_set_sizes, model_family,
                                                                    model_size,
                                                                    model_type, layer)

    # TTPDTest.from_data(acts_centered, acts, labels, polarities)


    ray.init(ignore_reinit_error=True)

    # Hyperparameter search space
    search_space = {
        # Logistic Regression hyperparameters
        'C': tune.loguniform(1e-3, 10),
        'fit_intercept': tune.choice([True, False]),
        'solver': tune.choice(['lbfgs', 'saga', 'liblinear']),

        # Polynomial features
        'degree': tune.choice([1, 2, 3]),
        'include_bias': tune.choice([True, False]),

        # Scaling options
        'use_scaler': tune.choice([True, False]),

        # Feature / interaction combinations
        'interaction_features': tune.choice([
            # main effects only
            ['proj_t_g', 'proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p'],
            ['proj_t_g', 'proj_t_p', 'inter1'],

            # simple pairwise interactions
            ['proj_t_g', 'proj_t_p', 'proj_t_g+proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_t_g-proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_t_g*proj_t_p'],

            # interactions with polarity
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_g*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_p*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'proj_t_g*proj_t_p*proj_p'],

            # include additive interaction term from TTPDTest
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1', 'proj_t_g*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1', 'proj_t_p*proj_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1', 'proj_t_g*proj_t_p*proj_p'],

            # higher-order combos
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1', 'proj_t_g*proj_p', 'proj_t_g*proj_t_p'],
            ['proj_t_g', 'proj_t_p', 'proj_p', 'inter1', 'proj_t_g+proj_t_p', 'proj_t_p*proj_p']
        ])
    }

    # Run Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_ttpd_test, acts_centered=acts_centered, acts=acts, labels=labels,
                             polarities=polarities),
        resources_per_trial={"cpu": 2},
        metric="accuracy",
        mode="max",
        num_samples=20,
        config=search_space,
        search_alg=BasicVariantGenerator()
    )

    print("Best config found: ", analysis.best_config)