import torch
import torch as t
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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
    # replace -1 with 0 to fit better onto a LogisticRegression
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(penalty=None, fit_intercept=True)
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    polarity_direc = LR_polarity.coef_
    return polarity_direc


class TTPDDT():
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
        proj_t_g = acts.numpy() @ self.t_g  # project onto general truth direction
        proj_p = acts.numpy() @ self.polarity_direc.T  # project into polarity dimension (affirm vs neg)
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d


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
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

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


class TTPD_DecisionTree():
    """
    Decision tree approach as described in Appendix B of "Truth is Universal".

    From the paper:
    "Not all statements are treated by the LLM as having either affirmative or
    negated polarity... it might seem natural to first categorise [a statement]
    as either affirmative or negated and then using a linear classifier based
    on t_affirmative or t_negated."

    This approach:
    1. Predicts whether the statement is affirmative or negated
    2. Uses the appropriate combined direction:
       - Affirmative: t_g + t_p (points from false to true)
       - Negated: t_g - t_p (points from false to true)
    3. Classifies based on projection onto the selected direction
    """

    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_classifier = None
        self.bias_affirmative = None
        self.bias_negated = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        """
        Train the decision tree approach.

        Args:
            acts_centered: Centered activations
            acts: Raw activations
            labels: Truth labels (+1 for true, -1 for false)
            polarities: Polarity labels (+1 for affirmative, -1 for negated)
        """
        probe = TTPD_DecisionTree()

        # Learn both truth directions
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy()

        # Train polarity classifier to predict affirmative vs negated
        polarities_copy = polarities.clone()
        polarities_copy[polarities_copy == -1.0] = 0.0
        probe.polarity_classifier = LogisticRegression(penalty=None, fit_intercept=True)
        probe.polarity_classifier.fit(acts.numpy(), polarities_copy.numpy())

        # Compute combined directions
        t_affirmative = probe.t_g + probe.t_p  # For affirmative statements
        t_negated = probe.t_g - probe.t_p  # For negated statements

        # Learn separate biases for affirmative and negated statements
        # by fitting on projections
        mask_affirmative = (polarities == 1.0)
        mask_negated = (polarities == -1.0)

        if mask_affirmative.sum() > 0:
            acts_aff = acts[mask_affirmative]
            labels_aff = labels[mask_affirmative]
            proj_aff = acts_aff.numpy() @ t_affirmative
            lr_aff = LogisticRegression(penalty=None, fit_intercept=True)
            lr_aff.fit(proj_aff[:, None], labels_aff.numpy())
            probe.bias_affirmative = lr_aff.intercept_[0]
        else:
            probe.bias_affirmative = 0.0

        if mask_negated.sum() > 0:
            acts_neg = acts[mask_negated]
            labels_neg = labels[mask_negated]
            proj_neg = acts_neg.numpy() @ t_negated
            lr_neg = LogisticRegression(penalty=None, fit_intercept=True)
            lr_neg.fit(proj_neg[:, None], labels_neg.numpy())
            probe.bias_negated = lr_neg.intercept_[0]
        else:
            probe.bias_negated = 0.0

        return probe

    def pred(self, acts):
        """
        Predict truth labels using the decision tree approach.

        Steps:
        1. Predict polarity (affirmative vs negated)
        2. Select appropriate direction (t_g + t_p or t_g - t_p)
        3. Project and classify
        """
        # Predict polarity for each statement
        pred_polarities = self.polarity_classifier.predict(acts.numpy())

        # Compute combined directions
        t_affirmative = self.t_g + self.t_p
        t_negated = self.t_g - self.t_p

        predictions = []
        for i, polarity in enumerate(pred_polarities):
            act = acts[i].numpy()

            if polarity == 1:  # Affirmative
                proj = act @ t_affirmative + self.bias_affirmative
            else:  # Negated
                proj = act @ t_negated + self.bias_negated

            # Classify based on projection (positive = true, negative = false)
            predictions.append(1.0 if proj > 0 else 0.0)

        return t.tensor(predictions)

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
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

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

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        LR = LogisticRegression(max_iter=5000, fit_intercept=True)

        grid = GridSearchCV(LR, param_grid, cv=5, n_jobs=-1)
        grid.fit(acts_4d, labels.numpy())

        # probe.LR.fit(actsS, labels.numpy())
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
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

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


class TTPD4dEnh3():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD4dEnh3()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

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
        acts_4d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()

        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        proj_p = proj_p.reshape(-1)

        if self.t_p is not None:
            proj_t_p = acts_np @ self.t_p
            proj_t_p = proj_t_p.reshape(-1)

            polarity_sign = np.sign(proj_p)

            polarity_sign[polarity_sign == 0] = 1.0

            aligned_tp = polarity_sign * proj_t_p

            inter = proj_t_g * aligned_tp

            acts_4d = np.stack([proj_t_g, aligned_tp, inter], axis=1)
        else:
            # TODO
            print("Should not reach here")
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d


class TTPD4dEnh2():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD4dEnh2()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

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
        acts_4d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()

        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        proj_p = proj_p.reshape(-1)

        if self.t_p is not None:
            proj_t_p = acts_np @ self.t_p
            proj_t_p = proj_t_p.reshape(-1)

            polarity_sign = np.sign(proj_p)

            polarity_sign[polarity_sign == 0] = 1.0

            aligned_tp = polarity_sign * proj_t_p

            inter = proj_t_g * aligned_tp
            sq = aligned_tp ** 2

            acts_4d = np.stack([proj_t_g, aligned_tp, inter, sq], axis=1)
        else:
            # TODO
            print("Should not reach here")
            acts_4d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_4d

# Extend the original implementation to use tP, and tP * p
class TTPD4dEnh():
    # Force LR to only use truth and polarity dimensions
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None
        self.scaler = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD4dEnh()
        # do a linear regression where X encodes truth.lie polarity, we ignore tP
        # Learn direction for truth
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy() if probe.t_p is not None else None

        # predict if the statement is affirmative or negated
        # gives a weight vector in activation space pointing towards affirmative vs negative phrasing

        # learn direction for polarity
        # project all activations into those 2 directions
        probe.polarity_direc = learn_polarity_direction(acts, polarities)

        # project all dimensions onto the 2d truth dimension (t_g and polarity)
        acts_4d = probe._project_acts(acts)

        # Scale features
        scaler = StandardScaler()
        actsS = scaler.fit_transform(acts_4d)
        probe.scaler = scaler

        probe.LR = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", fit_intercept=True)

        probe.LR.fit(actsS, labels.numpy())
        return probe

    def pred(self, acts):
        # same projection of all dimensions onto 3d
        acts_4d = self._project_acts(acts)

        if self.scaler:
            acts_4d = self.scaler.transform(acts_4d)
        return t.tensor(self.LR.predict(acts_4d))

    def _project_acts(self, acts):
        acts_np = acts.numpy()

        proj_t_g = acts_np @ self.t_g  # project onto general truth direction
        proj_p = acts_np @ self.polarity_direc.T

        proj_p = proj_p.reshape(-1)

        if self.t_p is not None:
            proj_t_p = acts_np @ self.t_p
            proj_t_p = proj_t_p.reshape(-1)

            polarity_sign = np.sign(proj_p)

            polarity_sign[polarity_sign == 0] = 1.0

            aligned_tp = polarity_sign * proj_t_p

            inter = proj_t_g * aligned_tp
            sq = aligned_tp ** 2

            acts_4d = np.stack([proj_t_g, aligned_tp, inter, sq], axis=1)
        else:
            # TODO
            print("Should not reach here")
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
TTPD_TYPES = [("TTPD", TTPD), ("TTPD4d", TTPD4d), ("TTPD4d1", TTPD4dEnh), ("TTPD4d2", TTPD4dEnh2), ("TTPD4d3", TTPD4dEnh3),
              ("TTPD2d", TTPD2d), ("TTPD3dTp", TTPD3dTp)
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
