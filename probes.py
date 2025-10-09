import torch
import torch as t
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


from utils import dataset_sizes, collect_training_data

# Cache
_polarity_params_cache = {}   # key: (shape, dtype) -> params dict
_ttpd_params_cache = {}       # key: (shape, dtype) -> params dict

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

# Define hyperparameter grid separately
param_gridp_pol_dir = {
    'lr__C': [0.01, 0.1],             # typical first test: low, medium, high regularization
    'lr__penalty': ['l1', 'l2'],       # test both common penalties
    'lr__solver': ['liblinear', 'saga'], # liblinear works for small datasets, saga can handle L1/L2/elasticnet
    'lr__max_iter': [1000]              # enough iterations to converge in most cases
}

def find_best_lr_params(X, y, param_grid=None, n_iter=15, random_state=42, scoring='balanced_accuracy'):
    """
    X: numpy array or torch tensor (n_samples, n_features)
    y: numpy array or torch tensor (n_samples,) with labels in {0,1}
    Returns: best_params dict (keys are sklearn LogisticRegression parameter names)
    """
    # convert tensors to numpy if needed
    if isinstance(X, t.Tensor):
        X_np = X.numpy()
    else:
        X_np = np.asarray(X)

    if isinstance(y, t.Tensor):
        y_np = y.numpy()
    else:
        y_np = np.asarray(y)

    if param_grid is None:
        param_grid = [
            {'lr__solver': ['liblinear'],
             'lr__penalty': ['l1', 'l2'],
             'lr__C': [0.01, 0.05, 0.1, 0.5, 1, 10],
             'lr__max_iter': [1000, 1500]},

            {'lr__solver': ['saga'],
             'lr__penalty': ['l1', 'l2'],
             'lr__C': [0.1, 1],
             'lr__max_iter': [1000, 2000, 3000]},

            {'lr__solver': ['saga'],
             'lr__penalty': ['elasticnet'],
             'lr__C': [0.1, 0.5, 1, 10],
             'lr__l1_ratio': [0.1, 0.3, 0.5],
             'lr__max_iter': [5000, 7000]}
        ]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(fit_intercept=True, max_iter=2000))
    ])

    # randomized search over pipeline's lr parameters
    rs = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=random_state
    )

    # catch convergence warnings, but let Grid search pick larger max_iter if needed
    rs.fit(X_np, y_np)


    # extract the best LR params (they are prefixed 'lr__' in rs.best_params_)
    best_params = {}
    for k, v in rs.best_params_.items():
        if k.startswith('lr__'):
            best_params[k[4:]] = v

    return best_params


def learn_truth_directions_new(acts_centered, labels, polarities, ridge=1e-4):
    """
    Returns t_g, t_p (torch tensors). If polarities are all zero or absent, returns (t_g, None).
    Inputs:
        acts_centered: torch tensor (n_samples, d_act)
        labels: torch tensor (n_samples,) with {0,1}
        polarities: torch tensor (n_samples,) with {-1, 1} or {0,1} (we convert)
    """
    # ensure float tensors
    acts_centered = acts_centered.to(dtype=t.float32)
    labels = labels.to(dtype=t.float32).clone()

    # normalize label space to signed {-1, +1} for the regression equation
    # expects labels in {0,1} originally; if they are already -1/1, this still works
    if labels.min() >= 0.0:
        y_signed = 2.0 * labels - 1.0   # 0->-1, 1->+1
    else:
        y_signed = labels

    # handle polarities: convert -1 -> -1, 1 -> +1, or 0 stays 0
    polarities = polarities.to(dtype=t.float32).clone()
    if polarities.min() >= 0.0 and polarities.max() <= 1.0:
        # convert {0,1} -> {-1,1} only if needed
        polarities_signed = 2.0 * polarities - 1.0
    else:
        polarities_signed = polarities

    # all zeros?
    all_polarities_zero = t.allclose(polarities_signed, t.zeros_like(polarities_signed), atol=1e-8)

    # build design matrix X
    if all_polarities_zero:
        X = y_signed.reshape(-1, 1)  # (n_samples, 1)
    else:
        # include interaction term y * polarity as second regressor
        X = t.column_stack([y_signed, y_signed * polarities_signed])  # (n_samples, 2)

    # solve (X^T X + ridge I) w = X^T acts_centered  --> stable ridge regression
    XT_X = X.T @ X
    d = XT_X.shape[0]
    reg = ridge * t.eye(d, dtype=XT_X.dtype)
    A = XT_X + reg
    B = X.T @ acts_centered  # (d, d_act)

    # solve for solution: A @ W = B  -> W = A^{-1} B
    # use torch.linalg.solve for numerical stability
    try:
        W = t.linalg.solve(A, B)  # (d, d_act)
    except RuntimeError:
        # fallback to pinv (safe)
        W = t.pinverse(A) @ B

    if all_polarities_zero:
        t_g = W.flatten()  # shape (d_act,)
        t_p = None
    else:
        t_g = W[0, :]
        t_p = W[1, :]

    return t_g, t_p

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


def learn_polarity_direction_hyper(acts, polarities, best_params=None, scoring='balanced_accuracy'):
    """
    acts: torch tensor (n_samples, d_act)
    polarities: torch tensor (n_samples,) with {-1,1} or {0,1}
    best_params: optional dict of LR params (to avoid search)
    """
    # unify shapes
    acts_np = acts.numpy() if isinstance(acts, t.Tensor) else np.asarray(acts)
    polar_np = polarities.numpy() if isinstance(polarities, t.Tensor) else np.asarray(polarities)

    key = (acts_np.shape, polar_np.shape, acts_np.dtype)

    # check cache
    if best_params is not None:
        lr_params = best_params
    else:
        # run search and cache result
        lr_params = find_best_lr_params(acts_np, polar_np, n_iter=10, scoring=scoring)

    # Build pipeline and fit (pipeline includes scaler)
    lr = LogisticRegression(fit_intercept=True, **lr_params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", lr)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipeline.fit(acts_np, polar_np)

    best_lr = pipeline.named_steps['lr']
    coef = best_lr.coef_  # shape (1, d_act) or (n_classes-1, d_act)
    # flatten to 1d (take first row)
    polarity_direc = np.ravel(coef[0])
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

        if TTPD2d.polarity_params is None:
            print("Set polarity parameters")
            TTPD2d.polarity_params = find_best_lr_params(acts, polarities)
            print("Params (Polarity): ", TTPD2d.polarity_params)

            # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities, TTPD2d.polarity_params)


        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        if TTPD2d.ttpd_params is None:
            print("Set ttpd parameters")
            TTPD2d.ttpd_params = find_best_lr_params(acts_4d, labels)
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
    polarity_params_cache = _polarity_params_cache
    ttpd_params_cache = _ttpd_params_cache


    def __init__(self, use_cache=True):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None         # numpy 1d vector
        self.pipeline = None               # sklearn Pipeline containing scaler+lr
        self.use_cache = use_cache

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities, scoring='balanced_accuracy'):
        """
        acts_centered: torch tensor (n_samples, d_act)
        acts: torch tensor (n_samples, d_act)
        labels: torch tensor (n_samples,) expected in {0,1}
        polarities: torch tensor (n_samples,) - in {-1,1} or {0,1}
        """
        probe = TTPD4dEnh()

        # ---- learn truth directions (regularized LS) ----
        probe.t_g, probe.t_p = learn_truth_directions_new(acts_centered, labels, polarities, ridge=1e-4)
        # make numpy arrays for projection
        probe.t_g = probe.t_g.numpy() if isinstance(probe.t_g, t.Tensor) else np.asarray(probe.t_g)
        probe.t_p = probe.t_p.numpy() if (probe.t_p is not None and isinstance(probe.t_p, t.Tensor)) else (probe.t_p if probe.t_p is None else np.asarray(probe.t_p))

        cache_key = (acts.shape, polarities.shape, acts.dtype)

        if probe.use_cache and cache_key in TTPD4dEnh.polarity_params_cache:
            polarity_params = TTPD4dEnh.polarity_params_cache[cache_key]
        else:
            print("Training Polarity params")
            polarity_params = find_best_lr_params(acts, polarities, n_iter=10, random_state=42, scoring=scoring)
            if probe.use_cache:
                TTPD4dEnh.polarity_params_cache[cache_key] = polarity_params
            print("Found: ", polarity_params)

        probe.polarity_direc = learn_polarity_direction_hyper(acts, polarities, best_params=polarity_params, scoring=scoring)

        # ---- project activations to our 4d representation ----
        acts_4d = probe._project_acts(acts)  # numpy array

        # ---- tune or reuse ttpd hyperparams keyed by acts_4d shape ----
        ttpd_cache_key = (acts_4d.shape, acts_4d.dtype)
        if probe.use_cache and ttpd_cache_key in TTPD4dEnh.ttpd_params_cache:
            ttpd_params = TTPD4dEnh.ttpd_params_cache[ttpd_cache_key]
        else:
            print("Training TTPD params")
            ttpd_params = find_best_lr_params(acts_4d, labels, n_iter=12, random_state=123, scoring=scoring)
            if probe.use_cache:
                TTPD4dEnh.ttpd_params_cache[ttpd_cache_key] = ttpd_params
            print("Found: ", ttpd_params)

        # ---- build final pipeline (keeps scaler for prediction) ----
        lr = LogisticRegression(fit_intercept=True, **ttpd_params)
        pipeline = Pipeline([("scaler", StandardScaler()), ("lr", lr)])

        # fit with convergence warnings suppressed; if convergence occurs, try increasing max_iter
        acts4_np = acts_4d if isinstance(acts_4d, np.ndarray) else np.asarray(acts_4d)
        labels_np = labels.numpy() if isinstance(labels, t.Tensor) else np.asarray(labels)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                pipeline.fit(acts4_np, labels_np)
        except Exception as e:
            # If solver struggles, try increasing max_iter and refit (robust fallback)
            lr.set_params(max_iter=5000)
            pipeline = Pipeline([("scaler", StandardScaler()), ("lr", lr)])
            pipeline.fit(acts4_np, labels_np)

        probe.pipeline = pipeline
        return probe

    def pred(self, acts):
        """
        acts: torch tensor or numpy array (n_samples, d_act)
        Returns predictions as torch tensor of {0,1}
        """
        acts_4d = self._project_acts(acts)
        # use pipeline.predict to ensure scaler is applied
        preds = self.pipeline.predict(acts_4d)
        return t.tensor(preds, dtype=t.int64)

    def _project_acts(self, acts):
        """
        Projects acts into [proj_t_g, proj_t_p, proj_p, proj_p * proj_t_p] (numpy array)
        Ensures consistent vector shapes and flattens polarity projection to 1D.
        """
        acts_np = acts.numpy() if isinstance(acts, t.Tensor) else np.asarray(acts)
        # ensure t_g and polarity vectors exist
        proj_t_g = acts_np @ self.t_g   # shape (n_samples,)
        # polarity_direc is 1d vector
        proj_p = acts_np @ self.polarity_direc   # shape (n_samples,)

        if self.t_p is not None:
            proj_t_p = acts_np @ self.t_p  # shape (n_samples,)
            # interaction term
            interaction = proj_p * proj_t_p
            p2 = proj_p ** 2
            tp2 = proj_t_p ** 2
            int2 = interaction ** 2
            # acts_4d = np.column_stack([proj_t_g, proj_t_p, proj_p, interaction, p2, tp2, int2])
            acts_4d = np.column_stack([proj_t_g, p2, tp2, int2])

        else:
            acts_4d = np.column_stack([proj_t_g, proj_p])

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
        probe.t_g, probe.t_p = learn_truth_directions_new(acts_centered, labels, polarities)
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
        #("TTPD2d", TTPD2d),
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
