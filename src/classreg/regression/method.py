from enum import Enum


class RegressionMethod(Enum):
    """
    Regression method implemented by a processor.

    Attributes
    ----------
    LINEAR_REGRESSION
        Ordinary least squares.
    RIDGE
        Linear regression with L2 regularization.
    LASSO
        Linear regression with L1 regularization.
    ELASTIC_NET
        Linear regression with combined L1 and L2 regularization.
    SVR
        Support vector regressor.
    K_NEAREST_NEIGHBORS
        K-nearest neighbors regressor.
    DECISION_TREE
        Decision tree regressor.
    RANDOM_FOREST
        Random forest regressor.
    HIST_GRADIENT_BOOSTING
        Histogram-based gradient boosting regressor.
    EXTRA_TREES
        Extremely randomized trees regressor.
    GRADIENT_BOOSTING
        Gradient boosting regressor.
    ADA_BOOST
        AdaBoost regressor.
    HUBER
        Linear regression robust to outliers.
    QUANTILE
        Linear regression of a conditional quantile.
    POISSON
        Generalized linear model with a Poisson distribution.
    GAUSSIAN_PROCESS
        Gaussian process regressor.
    MULTILAYER_PERCEPTRON
        Multilayer perceptron regressor.
    LINEAR_SVR
        Linear support vector regressor.
    STOCHASTIC_GRADIENT_DESCENT
        Linear regressor trained by stochastic gradient descent.
    AUTOMATIC_RELEVANCE_DETERMINATION
        Bayesian linear regression with one precision per feature.
    BAYESIAN_RIDGE
        Bayesian ridge regression.
    EXTRA_TREE
        Single extremely randomized tree regressor.
    GAMMA
        Generalized linear model with a Gamma distribution.
    TWEEDIE
        Generalized linear model with a Tweedie distribution.
    ISOTONIC
        Monotonic regression on a single feature.
    LARS
        Least angle regression.
    LASSO_LARS
        Lasso fitted by least angle regression.
    LASSO_LARS_INFORMATION_CRITERION
        Lasso whose strength is chosen by an information criterion.
    NU_SVR
        Nu support vector regressor.
    ORTHOGONAL_MATCHING_PURSUIT
        Greedy sparse linear regression.
    PARTIAL_LEAST_SQUARES
        Partial least squares regression.
    RADIUS_NEIGHBORS
        Radius neighbors regressor.
    THEIL_SEN
        Linear regression robust to outliers, based on the spatial median.
    BAGGING
        Ensemble of copies fitted on random subsets.
    STACKING
        Ensemble whose final method learns from base predictions.
    VOTING
        Ensemble that averages the predictions of its methods.
    KERNEL_RIDGE
        Ridge regression in a kernel-induced feature space.
    BASELINE
        Feature-independent reference regressor.
    """

    LINEAR_REGRESSION = "LinearRegression"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    ELASTIC_NET = "ElasticNet"
    SVR = "SVR"
    K_NEAREST_NEIGHBORS = "KNearestNeighbors"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    HIST_GRADIENT_BOOSTING = "HistGradientBoosting"
    EXTRA_TREES = "ExtraTrees"
    GRADIENT_BOOSTING = "GradientBoosting"
    ADA_BOOST = "AdaBoost"
    HUBER = "Huber"
    QUANTILE = "Quantile"
    POISSON = "Poisson"
    GAUSSIAN_PROCESS = "GaussianProcess"
    MULTILAYER_PERCEPTRON = "MultilayerPerceptron"
    LINEAR_SVR = "LinearSVR"
    STOCHASTIC_GRADIENT_DESCENT = "StochasticGradientDescent"
    AUTOMATIC_RELEVANCE_DETERMINATION = "AutomaticRelevanceDetermination"
    BAYESIAN_RIDGE = "BayesianRidge"
    EXTRA_TREE = "ExtraTree"
    GAMMA = "Gamma"
    TWEEDIE = "Tweedie"
    ISOTONIC = "Isotonic"
    LARS = "Lars"
    LASSO_LARS = "LassoLars"
    LASSO_LARS_INFORMATION_CRITERION = "LassoLarsInformationCriterion"
    NU_SVR = "NuSVR"
    ORTHOGONAL_MATCHING_PURSUIT = "OrthogonalMatchingPursuit"
    PARTIAL_LEAST_SQUARES = "PartialLeastSquares"
    RADIUS_NEIGHBORS = "RadiusNeighbors"
    THEIL_SEN = "TheilSen"
    BAGGING = "Bagging"
    STACKING = "Stacking"
    VOTING = "Voting"
    KERNEL_RIDGE = "KernelRidge"
    BASELINE = "Baseline"
