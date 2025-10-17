from sklearn.pipeline import Pipeline
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting

TREE_BASES = (
    BaseDecisionTree,
    ForestClassifier,
    ForestRegressor,
    BaseGradientBoosting,
)

LINEAR_BASES = (LinearModel, LinearClassifierMixin)

def get_estimator(est):
    # Grid/Randomized/Halving search objects
    if hasattr(est, "best_estimator_"):
        return get_estimator(est.best_estimator_)
    # Pipeline or FeatureUnion expose final steps via steps / transformer_list
    if isinstance(est, Pipeline):
        # last step is a (name, estimator) tuple
        return get_estimator(est.steps[-1][1])
    if hasattr(est, "estimator"):  # e.g., CalibratedClassifierCV
        return get_estimator(est.estimator)
    return est

def get_pipe(est):
    if is_pipeline(est):
        return est[:-1]
    elif hasattr(est, "best_estimator_"):
        return est.best_estimator_[:-1]
    else:
        raise ValueError("estimator is not a pipeline")

def is_linear(est):
    est = get_estimator(est)
    return isinstance(est, LINEAR_BASES)

def is_tree(est):
    est = get_estimator(est)
    if isinstance(est, TREE_BASES):
        return True
    # ensembles expose base estimators through `estimators_`
    return hasattr(est, "estimators_") and all(isinstance(e, BaseDecisionTree) for e in est.estimators_)

def is_pipeline(est):
    return isinstance(est, Pipeline)
