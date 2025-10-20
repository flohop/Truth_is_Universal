import os

from ray import  tune
import torch.nn as nn
import torch.optim as optim
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.inspection import permutation_importance

import torch
import torch as t
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import dataset_sizes, collect_training_data, collect_training_data_tuner, DataManager, \
    plot_lr_feature_importance

project_dir = os.getcwd()

def polarity_direction_lr(acts, polarities, C=0.1, penalty='l2', solver='lbfgs', max_iter=2000):
    """Modified to accept hyperparameters"""
    polarities_copy = polarities.clone()
    polarities_copy[polarities_copy == -1.0] = 0.0
    LR_polarity = LogisticRegression(
        penalty=penalty,
        C=C,
        fit_intercept=True,
        solver=solver,
        max_iter=max_iter
    )
    LR_polarity.fit(acts.numpy(), polarities_copy.numpy())
    return LR_polarity

def learn_polarity_direction_optimal(acts, polarities):
    """Modified to accept hyperparameters"""
    polarity_LR = LogisticRegression(
        penalty='l1',
        solver='saga',
        C=0.00033481,
        max_iter=5000,
        fit_intercept=True
    )
    polarity_LR.fit(acts.numpy(), polarities.numpy())
    return polarity_LR.coef_


def learn_polarity_direction(acts, polarities, C=0.1, penalty='l2', solver='lbfgs', max_iter=2000):
    """Modified to accept hyperparameters"""
    polarity_direc = polarity_direction_lr(acts, polarities, C=C, penalty=penalty,
                                           solver=solver, max_iter=max_iter).coef_
    return polarity_direc


def measure_polarity_direction_lr(train_acts, train_polarities, valid_acts, valid_polarities):
    LR = polarity_direction_lr(train_acts, train_polarities)

    pred = LR.predict(valid_acts)
    num_diff = np.sum(~np.isclose(pred, valid_polarities))
    return 1 -  num_diff / len(valid_acts)


class TTPDTestConfigurable():
    """Modified TTPDTest class with configurable hyperparameters"""

    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None
        self.config = None
        self.scaler = None
        self.proj_acts = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities, config=None):
        """
        Create probe with configurable hyperparameters

        Args:
            config: Dictionary with keys:
                - polarity_C: Regularization for polarity LR
                - polarity_penalty: Penalty type for polarity LR
                - polarity_solver: Solver for polarity LR
                - polarity_max_iter: Max iterations for polarity LR
                - use_scaler: True/False
                - features: List of features to use, given as a list e.g. ["a", "b", "c"]

                - final_penalty: Penalty for final LR (None or 'l2')
                - final_C: Regularization for final LR (only if penalty is not None)
                - final_solver: Solver for final LR
                - final_max_iter: Max iterations for final LR
        """
        if config is None:
            config = {}

        probe = TTPDTestConfigurable()
        probe.config = config

        # Learn truth directions (assuming this function exists)
        # If learn_truth_directions has parameters, add them to config
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy()

        # Learn polarity direction with configurable params
        probe.polarity_direc = learn_polarity_direction(
            acts, polarities,
            C=config.get('polarity_C', 0.1),
            penalty=config.get('polarity_penalty', 'l2'),
            solver=config.get('polarity_solver', 'lbfgs'),
            max_iter=config.get('polarity_max_iter', 2000)
        )


        # Project acts and fit final LR with configurable params
        acts_2d = probe._project_acts(acts, used_features=config.get("features", []))

        if config.get('use_scaler', True):
            probe.scaler = StandardScaler()
            acts_2d = probe.scaler.fit_transform(acts_2d)


        final_penalty = config.get('final_penalty', None)
        if final_penalty is None:
            probe.LR = LogisticRegression(
                penalty=None,
                fit_intercept=True,
                solver=config.get('final_solver', 'lbfgs'),
                max_iter=config.get('final_max_iter', 1000)
            )
        else:
            probe.LR = LogisticRegression(
                penalty=final_penalty,
                C=config.get('final_C', 1.0),
                fit_intercept=True,
                solver=config.get('final_solver', 'lbfgs'),
                max_iter=config.get('final_max_iter', 1000)
            )

        probe.proj_acts = acts_2d
        probe.LR.fit(acts_2d, labels.numpy())
        return probe

    def pred(self, acts):
        acts_2d = self._project_acts(acts, self.config["features"])
        if self.config.get('use_scaler', True):
            acts_2d = self.scaler.fit_transform(acts_2d)
        return t.tensor(self.LR.predict(acts_2d))

    def _project_acts(self, acts, used_features):

        # Base directions
        proj_t_g = acts.numpy() @ self.t_g
        proj_p = acts.numpy() @ self.polarity_direc.T
        proj_t_p = (acts.numpy() @ self.t_p)
        proj_t_p_inter = proj_t_p * proj_p[:, 0]

        inter1 = proj_t_g + proj_t_p
        inter2 = proj_t_g - proj_t_p
        inter3 = proj_t_g + proj_t_p_inter
        inter4 = proj_t_g - proj_t_p_inter
        inter5 = proj_t_g * proj_t_p_inter
        inter6 = proj_t_g * proj_t_p

        exp1 = proj_t_g ** 2
        exp2 = proj_t_p ** 2
        exp3 = proj_t_p_inter ** 2


        all_features = {
            "proj_t_g": proj_t_g[:, None],
            "proj_t_p": proj_t_p[:, None],
            "proj_p": proj_p,
            "proj_t_p_inter": proj_t_p_inter[:, None],
            "inter1": inter1[:, None],
            "inter2": inter2[:, None],
            "inter3": inter3[:, None],
            "inter4": inter4[:, None],
            "inter5": inter5[:, None],
            "inter6": inter6[:, None],
            "exp1": exp1[:, None],
            "exp2": exp2[:, None],
            "exp3": exp3[:, None],
        }

        extracted = [feat for (name, feat) in all_features.items() if name in used_features]

        # Features have to be given in ready form

        acts_2d = np.concatenate((extracted), axis=1)
        return acts_2d


def train_ttpd_with_config(config, data_loader_fn, train_sets, val_sets,
                           model_family, model_size, model_type, layer):
    """
    Trainable function for Ray Tune

    Args:
        config: Hyperparameter configuration from Ray Tune
        data_loader_fn: Function to load data (should be collect_training_data)
        train_sets: List of training dataset names
        val_sets: List of validation dataset names
        model_family, model_size, model_type, layer: Model specifications
    """

    config.update({f"polarity_{k}": v for k, v in config.pop("polarity_combo").items()})
    config.update({f"final_{k}": v for k, v in config.pop("final_combo").items()})

    # Load training data
    train_set_sizes = dataset_sizes(train_sets)
    acts_centered, acts, labels, polarities = data_loader_fn(
        train_sets, train_set_sizes, model_family, model_size, model_type, layer
    )

    # Train probe with current config
    probe = TTPDTestConfigurable.from_data(acts_centered, acts, labels, polarities, config)

    # Evaluate on validation sets
    val_set_sizes = dataset_sizes(val_sets)
    val_acts_centered, val_acts, val_labels, val_polarities = data_loader_fn(
        val_sets, val_set_sizes, model_family, model_size, model_type, layer
    )

    predictions = probe.pred(val_acts)
    accuracy = (predictions == val_labels).float().mean().item()

    # Report metrics to Ray Tune
    tune.report({"accuracy": accuracy, "loss": 1 - accuracy})


# ============================================================================
# Ray Tune Setup with Cross-Validation
# ============================================================================

def train_ttpd_with_cv(config, data_loader_fn, all_train_sets,
                       model_family, model_size, model_type, layer, cv_folds=6):
    """
    Trainable function with cross-validation

    Performs k-fold CV by holding out pairs of datasets
    """

    # unpack combos
    config.update({f"polarity_{k}": v for k, v in config.pop("polarity_combo").items()})
    config.update({f"final_{k}": v for k, v in config.pop("final_combo").items()})

    all_train_sets = np.array(all_train_sets)
    accuracies = []

    # Cross-validation loop
    indices = np.arange(0, len(all_train_sets), 2)
    for i in indices[:cv_folds]:
        # Hold out one pair for validation
        cv_train_sets = np.delete(all_train_sets, [i, i + 1], axis=0)
        cv_val_sets = all_train_sets[[i, i + 1]]

        # Load training data
        train_set_sizes = dataset_sizes(cv_train_sets.tolist())
        acts_centered, acts, labels, polarities = data_loader_fn(
            cv_train_sets, train_set_sizes, model_family, model_size, model_type, layer
        )

        # Train probe
        probe = TTPDTestConfigurable.from_data(acts_centered, acts, labels, polarities, config)

        # Evaluate on held-out validation sets
        for val_set in cv_val_sets:
            dm = DataManager()
            dm.add_dataset(val_set, model_family, model_size, model_type, layer,
                           split=None, center=False, device='cpu')
            val_acts, val_labels = dm.data[val_set]

            predictions = probe.pred(val_acts)
            acc = (predictions == val_labels).float().mean().item()
            accuracies.append(acc)

            tune.report({
                "accuracy": acc,
                "loss": 1 - acc
            })

    # Report mean accuracy across CV folds
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    tune.report({
        "accuracy": mean_accuracy,
        "accuracy_std": std_accuracy,
        "loss": 1 - mean_accuracy
    })


def get_search_space():
    """
    Define the hyperparameter search space — only valid (penalty, solver) pairs.
    """
    # Valid polarity LR combos
    polarity_options = [
        {"penalty": "l2", "solver": "lbfgs"},
        {"penalty": "l2", "solver": "liblinear"},
        {"penalty": "l1", "solver": "liblinear"},
        {"penalty": "l1", "solver": "saga"},
        {"penalty": None, "solver": "lbfgs"},
    ]

    # Valid final LR combos
    final_options = [
        {"penalty": "l2", "solver": "lbfgs"},
        {"penalty": "l2", "solver": "liblinear"},
        {"penalty": "l1", "solver": "liblinear"},
        {"penalty": None, "solver": "lbfgs"},
    ]

    search_space = {
        # choose one valid combination each time
        "polarity_combo": tune.choice(polarity_options),
        "final_combo": tune.choice(final_options),

        # for l1
        # scalar hyperparams
        "polarity_C": tune.loguniform(1e-4, 10.0),
        "polarity_l1_ratio": tune.uniform(0.0, 1.0),  # used only if elasticnet
        "polarity_max_iter": tune.choice([1000, 2000, 3000, 5000]),

        "final_C": tune.loguniform(1e-4, 10.0),
        "final_max_iter": tune.choice([500, 1000, 2000]),
        "final_l1_ratio": tune.uniform(0.0, 1.0),

        "use_scaler": tune.choice([True, False]),

        # feature sets
        "features": tune.choice([
            # 2 pairs
            # ["proj_t_g", "proj_p"],
            # ["proj_t_g", "exp2"],
            # ["proj_t_g", "proj_t_p"],
            # ["proj_t_g", "proj_p"],
            # ["proj_t_p", "proj_p"],
            # ["proj_t_g", "proj_t_p_inter"],
            # ["proj_t_g", "inter1"],
            # ["proj_t_g", "inter2"],
            # ["proj_t_g", "exp1"],
            # ["proj_t_g", "inter3"],

            # 3
            ["proj_t_g", "proj_p", "proj_t_p_inter"]
            # ["proj_t_g", "proj_t_p", "proj_p"],
            # ["exp1", "exp2", "exp3"],
            # ["proj_t_g", "exp2", "exp3"],
            # ["exp1", "exp2", "proj_t_p_inter"],
            # ["exp1", "proj_t_p", "exp2"],
            # ["proj_t_g", "exp2", "exp3", "inter1"],
            # ["proj_t_p", "inter1", "inter2"],

            # 4
            #["proj_t_g", "inter1", "inter2", "inter3"],
            # ["exp1", "exp2", "exp3", "inter1"],
            # ["exp1", "proj_t_p", "proj_p", "inter2"],
            # ["proj_t_g", "exp2", "proj_p", "inter3"],
            # ["proj_t_g", "proj_t_p", "exp3", "inter4"],
            # ["proj_p", "inter4", "inter5", "inter6"],

           # 5
            #["proj_t_g", "proj_t_p", "proj_p", "exp1", "exp2", "exp3"],
            # ["proj_t_p", "proj_p", "inter1", "exp1"],
            # ["proj_t_g", "proj_p", "inter1", "exp2"],
            # ["proj_t_g", "proj_t_p", "inter1", "exp3"],
            # ["proj_t_g", "proj_t_p", "proj_p", "inter2", "exp1"],
        ]),
    }

    return search_space

def optimize_ttpd_hyperparameters(
        data_loader_fn,
        train_sets,
        val_sets,
        model_family,
        model_size,
        model_type,
        layer,
        num_samples=100,
        max_concurrent_trials=4,
        use_cv=True,
        cv_folds=6,
        grace_period=5,
        use_gpu=False
):
    """
    Run hyperparameter optimization using Ray Tune

    Args:
        data_loader_fn: Function to load data (collect_training_data)
        train_sets: List of training dataset names
        val_sets: List of validation dataset names (ignored if use_cv=True)
        model_family, model_size, model_type, layer: Model specifications
        num_samples: Number of hyperparameter configurations to try
        max_concurrent_trials: Maximum number of parallel trials
        use_cv: Whether to use cross-validation
        cv_folds: Number of CV folds (pairs of datasets)
        grace_period: Minimum number of epochs before early stopping
        use_gpu: Whether to use GPU for training

    Returns:
        analysis: Ray Tune ExperimentAnalysis object with results
    """

    # Configure search algorithm (Optuna with TPE sampler)
    search_alg = OptunaSearch(
        metric="accuracy",
        mode="max"
    )

    # Configure scheduler (ASHA for early stopping of poor trials)
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=100,  # Maximum number of epochs
        grace_period=grace_period,
        reduction_factor=2
    )

    # Get search space
    search_space = get_search_space()

    # Configure resources per trial
    resources_per_trial = {"cpu": 1}
    if use_gpu:
        resources_per_trial["gpu"] = 0.25

    # Choose training function
    if use_cv:
        trainable = tune.with_parameters(
            train_ttpd_with_cv,
            data_loader_fn=data_loader_fn,
            all_train_sets=train_sets,
            model_family=model_family,
            model_size=model_size,
            model_type=model_type,
            layer=layer,
            cv_folds=cv_folds
        )
    else:
        trainable = tune.with_parameters(
            train_ttpd_with_config,
            data_loader_fn=data_loader_fn,
            train_sets=train_sets,
            val_sets=val_sets,
            model_family=model_family,
            model_size=model_size,
            model_type=model_type,
            layer=layer
        )

    # Run optimization
    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=num_samples,
        search_alg=search_alg,
        scheduler=scheduler,
        resources_per_trial=resources_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        verbose=1,
        name="ttpd_optimization"
    )

    return analysis


def analyze_results(analysis):
    """
    Analyze and print optimization results

    Args:
        analysis: Ray Tune ExperimentAnalysis object
    """
    # Get best configuration
    best_config = analysis.get_best_config(metric="accuracy", mode="max")
    best_trial = analysis.get_best_trial(metric="accuracy", mode="max")

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nBest Trial: {best_trial}")
    print(f"Best Accuracy: {best_trial.last_result['accuracy']:.4f}")

    if 'accuracy_std' in best_trial.last_result:
        print(f"Accuracy Std Dev: {best_trial.last_result['accuracy_std']:.4f}")

    print("\nBest Hyperparameters:")
    print("-" * 80)
    for param, value in best_config.items():
        print(f"  {param}: {value}")

    # Get dataframe with all results
    df = analysis.results_df

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    top_10 = df.nlargest(10, 'accuracy')[['accuracy', 'config/polarity_C',
                                          'config/polarity_penalty',
                                          'config/final_penalty']]
    print(top_10.to_string())

    # Parameter importance analysis
    print("\n" + "=" * 80)
    print("PARAMETER STATISTICS")
    print("=" * 80)

    for param in ['polarity_C', 'polarity_penalty', 'final_penalty']:
        if f'config/{param}' in df.columns:
            print(f"\n{param}:")
            print(df.groupby(f'config/{param}')['accuracy'].agg(['mean', 'std', 'count']))

    return best_config, df


def run_ray(train_sets, val_sets, num_samples=100):
    """
    Example of how to use the optimization pipeline
    """

    # train_sets = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors", "animal_class",
    #               "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts"]

    # val_sets = ["cities_conj", "cities_disj", "sp_en_trans_conj", "sp_en_trans_disj",
    #             "inventors_conj", "inventors_disj", "animal_class_conj", "animal_class_disj",
    #             "element_symb_conj", "element_symb_disj", "facts_conj", "facts_disj",
    #             "common_claim_true_false", "counterfact_true_false"]
    #
    # val_sets = ["cities_de", "neg_cities_de", "sp_en_trans_de", "neg_sp_en_trans_de", "inventors_de",
    #             "neg_inventors_de", "animal_class_de",
    #             "neg_animal_class_de", "element_symb_de", "neg_element_symb_de", "facts_de", "neg_facts_de"]

    # Model specifications
    model_family = 'Llama3'
    model_size = '8B'
    model_type = 'chat'
    layer = 12

    # Run optimization with cross-validation
    print("Starting hyperparameter optimization with Ray Tune...")
    analysis = optimize_ttpd_hyperparameters(
        data_loader_fn=collect_training_data_tuner,
        train_sets=train_sets,
        val_sets=val_sets,
        model_family=model_family,
        model_size=model_size,
        model_type=model_type,
        layer=layer,
        num_samples=num_samples,
        max_concurrent_trials=4,  # Run 4 trials in parallel
        use_cv=True,  # Use cross-validation
        cv_folds=6,  # 6-fold CV
        grace_period=3,
        use_gpu=False
    )

    # Analyze results
    best_config, results_df = analyze_results(analysis)

    # Train final model with best hyperparameters
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)

    train_set_sizes = dataset_sizes(train_sets)
    acts_centered, acts, labels, polarities = collect_training_data(
        train_sets, train_set_sizes, model_family, model_size, model_type, layer
    )

    final_probe = TTPDTestConfigurable.from_data(
        acts_centered, acts, labels, polarities, best_config
    )

    # Evaluate on validation sets
    val_set_sizes = dataset_sizes(val_sets)
    val_acts_centered, val_acts, val_labels, val_polarities = collect_training_data(
        val_sets, val_set_sizes, model_family, model_size, model_type, layer
    )

    predictions = final_probe.pred(val_acts)
    final_accuracy = (predictions == val_labels).float().mean().item()

    print(f"\nFinal Model Validation Accuracy: {final_accuracy:.4f}")

    return final_probe, best_config, analysis

class TTPDOptimal():
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction_optimal(acts, polarities)
        acts_2d = probe._project_acts(acts)
        probe.LR = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            C=0.00560875,
            max_iter=1000)

        probe.LR.fit(acts_2d, labels.numpy())
        return probe

    def pred(self, acts):
        acts_2d = self._project_acts(acts)
        return t.tensor(self.LR.predict(acts_2d))

    def _project_acts(self, acts):
        proj_t_g = acts.numpy() @ self.t_g
        proj_t_p = (acts.numpy() @ self.t_p)
        acts_2d = np.concatenate((proj_t_g[:, None], proj_t_p[:, None]), axis=1)
        return acts_2d


class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direc = None
        self.LR = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPD()
        probe.t_g, _ = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)
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

    # --- NEW METHOD ---
    def get_intervention_vector(self):
        """
        Calculates the high-dimensional intervention vector from the probe.

        This vector reconstructs the optimal direction from the 2D probe space
        back into the model's high-dimensional activation space.

        Returns:
            torch.Tensor: The intervention vector with shape (d_model,).
        """
        if self.t_g is None or self.LR is None:
            raise RuntimeError("Probe is not trained. Call TTPD.from_data() first.")

        # 1. Extract the basis vectors (the directions you projected onto)
        basis_vector_1 = t.tensor(self.t_g)
        basis_vector_2 = t.tensor(self.polarity_direc.flatten())

        # 2. Extract the logistic regression coefficients from the 2D probe
        lr_coefs = self.LR.coef_.flatten()
        c1, c2 = lr_coefs[0], lr_coefs[1]

        # 3. Calculate the final vector as a weighted sum of the basis vectors
        intervention_vector = (c1 * basis_vector_1) + (c2 * basis_vector_2)

        return intervention_vector

    # --- NEW METHOD ---
    def save_intervention_vector(self, filepath, dtype=t.bfloat16):
        """
        Calculates and saves the high-dimensional intervention vector to a file.

        Args:
            filepath (str): The path to save the .pt file (e.g., "TTPD_probe.pt").
            dtype (torch.dtype, optional): The dtype for model compatibility.
                                           Defaults to torch.bfloat16.
        """
        # Get the correctly shaped vector
        intervention_vector = self.get_intervention_vector()

        # Cast to the desired dtype to match your model
        intervention_vector = intervention_vector.to(dtype)

        # Save the tensor
        t.save(intervention_vector, filepath)
        print(f"✅ Intervention vector with shape {intervention_vector.shape} saved to {filepath}")

class TTPDFeatureImp():
    def __init__(self):
        self.t_g = None
        self.t_p = None
        self.polarity_direc = None
        self.LR = None
        self.LR_norm = None

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities):
        probe = TTPDFeatureImp()
        probe.t_g, probe.t_p = learn_truth_directions(acts_centered, labels, polarities)
        probe.t_g = probe.t_g.numpy()
        probe.t_p = probe.t_p.numpy()

        probe.polarity_direc = learn_polarity_direction(acts, polarities)
        acts_2d = probe._project_acts(acts)

        scaler = StandardScaler()
        acts_2d_scaled = scaler.fit_transform(acts_2d)

        probe.LR_norm = LogisticRegression(penalty=None, fit_intercept=True)

        probe.LR_norm.fit(acts_2d_scaled, labels)

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


def learn_truth_directions(acts_centered, labels, polarities):
    all_polarities_zero = t.allclose(polarities, t.tensor([0.0]), atol=1e-8)
    labels_copy = t.where(labels == 0, t.tensor(-1.0), labels)
    X = labels_copy.reshape(-1,1) if all_polarities_zero else t.column_stack([labels_copy, labels_copy*polarities])
    solution = t.linalg.inv(X.T @ X) @ X.T @ acts_centered
    if all_polarities_zero:
        return solution.flatten(), None
    else:
        return solution[0,:], solution[1,:]

# def learn_polarity_direction(acts, polarities):
#     polarities_copy = polarities.clone()
#     polarities_copy[polarities_copy == -1] = 0
#     LR = LogisticRegression(penalty='l2', C=0.1, fit_intercept=True, solver='lbfgs', max_iter=2000)
#     LR.fit(acts.numpy(), polarities_copy.numpy())
#     return LR.coef_


class MLPProbe(nn.Module):
    """
    A non-linear probe using a simple Multi-Layer Perceptron.
    This can learn a more complex decision boundary than logistic regression.
    """

    def __init__(self, input_dim, hidden_dim=512):
        super(MLPProbe, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_linear_probe_cv_and_save(X, y, model_dtype, device):
    """
    Trains a robust linear probe using LogisticRegressionCV to find the optimal
    regularization strength and saves its coefficient vector.
    """
    print("\nSplitting data into training and testing sets...")
    # Using numpy for sklearn compatibility
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    print(f"Training robust linear probe with Cross-Validation on {X_train.shape[0]} samples...")

    # Use LogisticRegressionCV to automatically find the best regularization strength (C).
    # This is more robust than picking a single C value.
    probe = LogisticRegressionCV(
        Cs=100,  # Try 10 different C values
        cv=5,  # Use 5-fold cross-validation
        penalty='l1',
        solver='saga',
        max_iter=2000,  # Increase if it doesn't converge
        random_state=42,
        tol=0.01,
        n_jobs=-1  # Use all available CPU cores
    )

    probe.fit(X_train, y_train)

    # Evaluate the probe's performance
    accuracy = probe.score(X_test, y_test)
    print(f"\nProbe Accuracy on test set: {accuracy:.4f}")
    print(f"Best regularization strength found (C): {probe.C_[0]}")

    # Extract the learned coefficient vector (shape is 1, 4096)
    truth_vector = t.tensor(probe.coef_.flatten(), device=device, dtype=model_dtype)

    # Save the vector for use in intervention scripts
    save_path = "linear_cv_probe.pt"
    t.save(truth_vector, save_path)
    print(f"Successfully saved the truth vector (shape: {truth_vector.shape}) to '{save_path}'")

    num_zeroed = np.sum(probe.coef_ == 0)
    print(f"L1 regularization zeroed out {num_zeroed} / {X_train.shape[1]} features.")

def train_mlp_probe_and_save_vector(X, y, model_dtype, device, epochs=50, batch_size=64):
    """
    Trains an MLP probe using PyTorch and saves the weight of its final layer.
    """
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.cpu()
    )

    input_dim = X_train.shape[1]
    probe = MLPProbe(input_dim).to(device)

    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.AdamW(probe.parameters(), lr=0.0001, weight_decay=0.01)

    print(f"Training MLP probe on {X_train.shape[0]} samples...")

    probe.train()
    for epoch in range(epochs):
        permutation = t.randperm(X_train.size(0))
        for i in tqdm(range(0, X_train.size(0), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            indices = permutation[i:i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = probe(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate the probe's performance
    probe.eval()
    with t.no_grad():
        y_pred_tensor = probe(X_test.to(device))
        y_pred = (y_pred_tensor > 0.5).float().cpu().numpy()
        accuracy = accuracy_score(y_test.cpu().numpy(), y_pred)

    print(f"\nMLP Probe Accuracy on test set: {accuracy:.4f}")

    # Extract the weight vector from the *output layer* of the MLP.
    # This is a good approximation of the final learned "truth direction".
    truth_vector = probe.output_layer.weight.data.flatten().to(device).to(model_dtype)

    # Save the vector for intervention scripts
    save_path = "mlp_probe.pt"
    t.save(truth_vector, save_path)
    print(f"Successfully saved the MLP probe's final layer weight to '{save_path}'")



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

        if labels is not None:  # flip direction if needed
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


def train_probe_and_save_vector(X, y, model_dtype, device):
    """
    Trains a logistic regression probe and saves its coefficient vector.
    """
    print("\nSplitting data into training and testing sets...")
    # Using numpy for sklearn compatibility
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )

    print(f"Training probe on {X_train.shape[0]} samples...")

    # Define the logistic regression model with strong L1 regularization
    # L1 (Lasso) is excellent for high-dimensional data as it promotes sparsity.
    # A small C means strong regularization. This is crucial to prevent overfitting.
    probe = LogisticRegression(
        penalty='l1',
        C=0.01,
        solver='saga',
        max_iter=100,  # Increase if it doesn't converge
        random_state=42,
        tol=0.01  # Looser tolerance for faster convergence
    )

    probe.fit(X_train, y_train)

    # Evaluate the probe's performance
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nProbe Accuracy on test set: {accuracy:.4f}")

    # Extract the learned coefficient vector
    # This vector represents the direction of "truth" in the 4096-dim space
    truth_vector = t.tensor(probe.coef_.flatten(), device=device, dtype=model_dtype)

    # Save the vector for use in intervention scripts
    save_path = "high_dim_probe.pt"
    t.save(truth_vector, save_path)
    print(f"Successfully saved the truth vector to '{save_path}'")

    num_zeroed = np.sum(probe.coef_ == 0)
    print(f"L1 regularization zeroed out {num_zeroed} / {X_train.shape[1]} features.")


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
        pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean
        # project activations onto direction
        proj = acts @ direction
        # fit bias
        LR = LogisticRegression(penalty=None, fit_intercept=True)
        LR.fit(proj[:, None], labels)

        probe = MMProbe(direction, LR).to(device)

        return probe


TTPD_TYPES = [("TTPD", TTPD), ("TTPDOpt", TTPDOptimal)]
ALL_PROBES = TTPD_TYPES + [("CSSProbe", CCSProbe), ("LRProbe", LRProbe)]

def get_average_feature_importance(t_acts_centered, t_acts, t_labels, t_polarities, config, runs=10):
    total_imp = None
    for _ in tqdm(range(runs), desc="Averaging coefficients"):
        ttpd = TTPDTestConfigurable.from_data(
            t_acts_centered, t_acts, t_labels, t_polarities, config=config
        )
        imp_mean = permutation_importance(ttpd.LR, ttpd.proj_acts, t_labels)["importances_mean"]
        if total_imp is None:
            total_imp = imp_mean
        else:
            total_imp += imp_mean
    return total_imp / float(runs)



# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == '__main__':
    model_family = 'Llama3'
    model_size = '8B'
    model_type = 'chat'
    layer = 12
    device = 'mps' if torch.backends.mps.is_built() else 'cpu'

    # import ray
    #
    # ray.init(num_cpus=4, ignore_reinit_error=True)
    #
    # # Run optimization
    # final_probe, best_config, analysis = run_ray()
    #
    # # Shutdown Ray
    # ray.shutdown()

    # example_config = {
    #     # Polarity logistic regression
    #     "polarity_C": 0.5,
    #     "polarity_penalty": "l2",
    #     "polarity_solver": "lbfgs",
    #     "polarity_max_iter": 2000,
    #
    #     # Features used by the model
    #     "features": ["proj_t_g", "proj_t_p", "proj_p", "proj_t_p_inter", "inter1", "inter2", "inter3", "inter4" "inter5", "inter6", "exp1", "exp2", "exp3"],
    #
    #     # Final logistic regression
    #     "final_penalty": "l2",
    #     "final_C": 1.0,
    #     "final_solver": "lbfgs",
    #     "final_max_iter": 1000,
    # }
    #
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

    t_acts_centered, t_acts, t_labels, t_polarities = collect_training_data(val_sets, dataset_sizes(val_sets),
                                                                            model_family, model_size, model_type, layer)


    avg_coef, avg_coef_norm = get_average_feature_importance(acts_centered_train, acts_train, labels_train, polarities_train, runs=100)

    plot_lr_feature_importance(avg_coef, feature_names=["t_g_proj", "p_proj"])
    plot_lr_feature_importance(avg_coef, feature_names=["t_g_proj", "p_proj"], title="Feature Importance Normalized (|Coefficient|)")






