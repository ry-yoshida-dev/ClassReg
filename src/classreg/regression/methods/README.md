# methods

## Overview

Parameter classes and enums for each regression method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [linear_regression/](./linear_regression/README.md) | Ordinary least squares parameters. |
| [ridge/](./ridge/README.md) | Ridge parameters. |
| [lasso/](./lasso/README.md) | Lasso parameters. |
| [elastic_net/](./elastic_net/README.md) | Elastic net parameters. |
| [svr/](./svr/README.md) | Support vector regression parameters, kernels, and gamma rules. |
| [linear_svr/](./linear_svr/README.md) | Linear support vector regression parameters and losses. |
| [k_nearest_neighbors/](./k_nearest_neighbors/README.md) | K-nearest neighbors parameters and related enums. |
| [decision_tree/](./decision_tree/README.md) | Decision tree parameters. |
| [random_forest/](./random_forest/README.md) | Random forest parameters and feature sampling rules. |
| [hist_gradient_boosting/](./hist_gradient_boosting/README.md) | Histogram gradient boosting parameters and losses. |
| [extra_trees/](./extra_trees/README.md) | Extremely randomized trees parameters. |
| [gradient_boosting/](./gradient_boosting/README.md) | Gradient boosting parameters and losses. |
| [ada_boost/](./ada_boost/README.md) | AdaBoost parameters and losses. |
| [huber/](./huber/README.md) | Huber regression parameters. |
| [quantile/](./quantile/README.md) | Quantile regression parameters. |
| [poisson/](./poisson/README.md) | Poisson regression parameters. |
| [gaussian_process/](./gaussian_process/README.md) | Gaussian process parameters and kernels. |
| [multilayer_perceptron/](./multilayer_perceptron/README.md) | Multilayer perceptron parameters. |
| [stochastic_gradient_descent/](./stochastic_gradient_descent/README.md) | Stochastic gradient descent parameters and losses. |
| [automatic_relevance_determination/](./automatic_relevance_determination/README.md) | Automatic relevance determination parameters. |
| [bayesian_ridge/](./bayesian_ridge/README.md) | Bayesian ridge parameters. |
| [extra_tree/](./extra_tree/README.md) | Single extremely randomized tree parameters. |
| [gamma/](./gamma/README.md) | Gamma regression parameters. |
| [tweedie/](./tweedie/README.md) | Tweedie regression parameters and links. |
| [isotonic/](./isotonic/README.md) | Isotonic regression parameters, directions, and out-of-bounds rules. |
| [lars/](./lars/README.md) | Least angle regression parameters. |
| [lasso_lars/](./lasso_lars/README.md) | Lasso least angle regression parameters. |
| [lasso_lars_ic/](./lasso_lars_ic/README.md) | Lasso with information criterion parameters and criteria. |
| [nu_svr/](./nu_svr/README.md) | Nu support vector regression parameters, kernels, and gamma rules. |
| [orthogonal_matching_pursuit/](./orthogonal_matching_pursuit/README.md) | Orthogonal matching pursuit parameters. |
| [partial_least_squares/](./partial_least_squares/README.md) | Partial least squares parameters. |
| [radius_neighbors/](./radius_neighbors/README.md) | Radius neighbors parameters and related enums. |
| [theil_sen/](./theil_sen/README.md) | Theil-Sen parameters. |
| [bagging/](./bagging/README.md) | Bagging parameters. |
| [stacking/](./stacking/README.md) | Stacking parameters. |
| [voting/](./voting/README.md) | Voting parameters. |
| [kernel_ridge/](./kernel_ridge/README.md) | Kernel ridge parameters. |
| [baseline/](./baseline/README.md) | Feature-independent reference regressor. |
