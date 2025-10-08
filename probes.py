import torch
import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from utils import dataset_sizes, collect_training_data

param_grid = [
    # liblinear: supports L1, L2
    {'solver': ['liblinear'],
     'penalty': ['l1', 'l2'],
     'C': [0.1, 1, 10, 100]},

    # lbfgs: only supports L2
    {'solver': ['lbfgs'],
     'penalty': ['l2'],
     'C': [0.1, 1, 10, 100]},

    # saga: supports L1, L2, elasticnet
    {'solver': ['saga'],
     'penalty': ['l1', 'l2'],
     'C': [0.1, 1, 10, 100]},

    # saga + elasticnet with l1_ratio
    {'solver': ['saga'],
     'penalty': ['elasticnet'],
     'l1_ratio': [0.1, 0.5, 0.9],
     'C': [0.01, 0.1, 1, 10, 100]}
]

param_grid_pipeline = [
    {'lr__solver': ['liblinear'],
     'lr__penalty': ['l1', 'l2'],
     'lr__C': [0.001, 0.1, 1, 10],
     'lr__max_iter': [1000, 2000]},  # increase max_iter

    {'lr__solver': ['lbfgs'],
     'lr__penalty': ['l2'],
     'lr__C': [0.001, 0.1, 1, 10],
     'lr__max_iter': [1000, 2000]},

    {'lr__solver': ['saga'],
     'lr__penalty': ['l1', 'l2'],
     'lr__C': [0.001, 0.1, 1, 10],
     'lr__max_iter': [1000, 2000]},

    {'lr__solver': ['saga'],
     'lr__penalty': ['elasticnet'],
     'lr__l1_ratio': [0.1, 0.5, 0.9],
     'lr__C': [0.001, 0.01, 0.1, 1, 10],
     'lr__max_iter': [2000]}  # elasticnet often needs more iterations
]


# Define your pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression())
])

# Define hyperparameter grid separately
param_gridp_pol_dir = {
    'lr__C': [0.01, 0.1],             # typical first test: low, medium, high regularization
    'lr__penalty': ['l1', 'l2'],       # test both common penalties
    'lr__solver': ['liblinear', 'saga'], # liblinear works for small datasets, saga can handle L1/L2/elasticnet
    'lr__max_iter': [1000]              # enough iterations to converge in most cases
}


def find_best_lr_params(X, y, param_grid=None, n_iter=15, random_state=42):
    if param_grid is None:
        param_grid = [
            # liblinear solver
            {'lr__solver': ['liblinear'],
             'lr__penalty': ['l1', 'l2'],
             'lr__C': [0.01, 0.1, 0.5, 1, 10],
             'lr__max_iter': [1000, 1500]},

            # saga solver with l1 or l2 penalty
            {'lr__solver': ['saga'],
             'lr__penalty': ['l1', 'l2'],
             'lr__C': [0.1, 1],
             'lr__max_iter': [1000, 2000, 3000]},

            # saga solver with elasticnet
            {'lr__solver': ['saga'],
             'lr__penalty': ['elasticnet'],
             'lr__C': [0.01, 0.1, 1, 10],
             'lr__l1_ratio': [0.1, 0.5, 0.9],
             'lr__max_iter': [1000, 2000]}
        ]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(fit_intercept=True))
    ])

    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=random_state
    )

    grid_search.fit(X, y)
    # Extract only the LR step parameters
    best_params = {}
    for key, value in grid_search.best_params_.items():
        if key.startswith('lr__'):
            best_params[key[4:]] = value

    return best_params


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


def learn_polarity_direction(acts, polarities):
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc


def learn_polarity_direction_hyper(acts, polarities, best_params=None):
    """
    Learns the polarity direction using LogisticRegression.
    If best_params is provided, it uses them directly instead of running grid search.

    Parameters:
        acts: tensor of activations
        polarities: tensor of polarities (-1 or 1)
        best_params: dict of best LogisticRegression parameters (optional)

    Returns:
        polarity_direc: numpy array of learned coefficients
    """
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0

    # If best_params are given, skip grid search
    if best_params is not None:
        lr = LogisticRegression(fit_intercept=True, **best_params)
        pipeline = Pipeline([
            # ("scaler", StandardScaler()),
            ("lr", lr)
        ])
        pipeline.fit(acts.numpy(), polarities_copy.numpy())
        best_lr = pipeline.named_steps['lr']
    else:
        grid_search = RandomizedSearchCV(
            estimator=Pipeline([
                # ("scaler", StandardScaler()),
                ("lr", LogisticRegression(fit_intercept=True)),
            ]),
            param_distributions=param_gridp_pol_dir,
            n_iter=10,
            scoring="accuracy",
            cv=3,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )
        grid_search.fit(acts.numpy(), polarities_copy.numpy())
        best_lr = grid_search.best_estimator_.named_steps['lr']

    polarity_direc = best_lr.coef_
    return polarity_direc


# Extend the original implementation to use tP * p
class TTPD3dTpInv():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD3dTpInv()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_3d = probe._project_acts(acts)

        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)

        probe.LR.fit(acts_3d, labels.numpy())
        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_3d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_3d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()
        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        if self.t_p is not None:
            proj_t_p_inter = (acts_np @ self.t_p) * proj_p[:, 0]
            acts_4d = np.concatenate((proj_t_g[:, None], proj_t_p_inter[:, None], proj_p), axis=1)
        else:
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d

# Extend the original implementation to use tP
class TTPD3dTp():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD3dTp()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        LR = LogisticRegression(max_iter=5000, fit_intercept=True)

        grid = GridSearchCV(LR, param_grid, cv=5, n_jobs=-1)
        grid.fit(acts_4d, labels.numpy())

        # probe.LR.fit(actsS, labels.numpy())
        probe.LR = grid.best_estimator_
        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_3d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_3d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()
        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        if self.t_p is not None:
            proj_t_p = (acts_np @ self.t_p)
            acts_3d = np.concatenate((proj_t_g[:, None], proj_t_p[:, None], proj_p), axis=1)
        else:
            acts_3d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_3d


# use only tG and tP * p
class TTPD2d():
    polarity_params = None
    ttpd_params = None

    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD2d()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        if TTPD4dEnh.polarity_params is None:
            print("Set polarity parameters")
            TTPD4dEnh.polarity_params = find_best_lr_params(acts, polarities)
            print("Params (Polarity): ", TTPD2d.polarity_params)

            # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities, TTPD2d.polarity_params)


        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        if TTPD4dEnh.ttpd_params is None:
            print("Set ttpd parameters")
            TTPD4dEnh.ttpd_params = find_best_lr_params(acts_4d, labels)
            print("Params (TTPD): ", TTPD2d.ttpd_params)

        LR = LogisticRegression(fit_intercept=True, **TTPD2d.ttpd_params)

        grid = GridSearchCV(LR, param_grid, cv=5, n_jobs=-1)
        grid.fit(acts_4d, labels.numpy())

        probe.LR = grid.best_estimator_
        return probe

        probe.LR.fit(acts_4d, labels.numpy())
        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_4d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()
        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        if self.t_p is not None:
            proj_t_p_inter = (acts_np @ self.t_p) * proj_p[:, 0]
            acts_4d = np.concatenate((proj_t_g[:, None], proj_t_p_inter[:, None]), axis=1)
        else:
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d


class TTPD4dEnh():
    polarity_params = None
    ttpd_params = None
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=UserWarning)
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD4dEnh()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        if TTPD4dEnh.polarity_params is None:
            print("Set polarity parameters")
            TTPD4dEnh.polarity_params = find_best_lr_params(acts, polarities)
            print("Params (Polarity): ", TTPD4dEnh.polarity_params)

        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities, TTPD4dEnh.polarity_params)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        if TTPD4dEnh.ttpd_params is None:
            print("Set ttpd parameters")
            TTPD4dEnh.ttpd_params = find_best_lr_params(acts_4d, labels)
            print("Params (TTPD): ", TTPD4dEnh.ttpd_params)

        lr = LogisticRegression(fit_intercept=True, **TTPD4dEnh.ttpd_params)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", lr),
        ])
        pipeline.fit(acts_4d, labels.numpy())
        probe.LR = pipeline.named_steps["lr"]

        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_4d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()
        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        if self.t_p is not None:
            proj_t_p = (acts_np @ self.t_p)
            acts_4d = np.concatenate((proj_t_g[:, None], proj_t_p[:, None], proj_p), axis=1)
        else:
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d


# Extend the original implementation to use tP, and tP * p
class TTPD4d():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD4d()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)

        probe.LR.fit(acts_4d, labels.numpy())
        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_4d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()
        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        if self.t_p is not None:
            proj_t_p = (acts_np @ self.t_p)
            acts_4d = np.concatenate((proj_t_g[:, None], proj_t_p[:, None], proj_p), axis=1)
        else:
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d


class TTPD():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_2d = probe._project_acts(acts)

        probe.LR = LogisticRegression(penalty=None, fit_intercept=True)

        # model only has to figure out truth/lie given these two features
        probe.LR.fit(acts_2d, labels.numpy())
        return probe
    
    def pred(self, acts):
        # same projection of all dimensions onto 2d
        acts_2d = self._project_acts(acts)

        # use prev trained LR using these 2 dimensions for predictions
        return t.tensor(self.LR.predict(acts_2d))
    
    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g # project onto general truth direction
        proj_p = acts.numpy() @ self.polarity_direc.T # project into polarity dimension (affirm vs neg)
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
        #("TTPD4d", TTPD4d),
        ("TTPD4dHyper", TTPD4dEnh),
        ("TTPD2d", TTPD2d),
    #  ("TTPD3dTp", TTPD3dTp)
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
    # get size of each training dataset to include an equal number of statements from each topic in training data
    train_set_sizes = dataset_sizes(train_sets)

    cv_train_sets = np.array(train_sets)
    acts_centered, acts, labels, polarities = collect_training_data(cv_train_sets, train_set_sizes, model_family,
                                                                    model_size, model_type, layer)

    # Make sure no code errors
    for (name, ttpd) in TTPD_TYPES:
        print(name)
        probe = ttpd.from_data(acts_centered, acts, labels, polarities)

        predictions = probe.pred(acts)
        print(predictions)
