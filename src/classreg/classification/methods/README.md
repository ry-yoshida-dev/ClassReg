# methods

## Overview

Parameter classes and enums for each classification method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [logistic_regression/](./logistic_regression/README.md) | Logistic regression parameters and solvers. |
| [svc/](./svc/README.md) | Support vector classification parameters, kernels, and gamma rules. |
| [linear_svc/](./linear_svc/README.md) | Linear support vector classification parameters, penalties, and losses. |
| [k_nearest_neighbors/](./k_nearest_neighbors/README.md) | K-nearest neighbors parameters and related enums. |
| [decision_tree/](./decision_tree/README.md) | Decision tree parameters. |
| [random_forest/](./random_forest/README.md) | Random forest parameters and feature sampling rules. |
| [hist_gradient_boosting/](./hist_gradient_boosting/README.md) | Histogram gradient boosting parameters. |
| [gaussian_naive_bayes/](./gaussian_naive_bayes/README.md) | Gaussian naive Bayes parameters. |
| [extra_trees/](./extra_trees/README.md) | Extremely randomized trees parameters. |
| [gradient_boosting/](./gradient_boosting/README.md) | Gradient boosting parameters. |
| [ada_boost/](./ada_boost/README.md) | AdaBoost parameters. |
| [linear_discriminant_analysis/](./linear_discriminant_analysis/README.md) | Linear discriminant analysis parameters and solvers. |
| [quadratic_discriminant_analysis/](./quadratic_discriminant_analysis/README.md) | Quadratic discriminant analysis parameters. |
| [multilayer_perceptron/](./multilayer_perceptron/README.md) | Multilayer perceptron parameters. |
| [stochastic_gradient_descent/](./stochastic_gradient_descent/README.md) | Stochastic gradient descent parameters and losses. |
| [bernoulli_naive_bayes/](./bernoulli_naive_bayes/README.md) | Bernoulli naive Bayes parameters. |
| [multinomial_naive_bayes/](./multinomial_naive_bayes/README.md) | Multinomial naive Bayes parameters. |
| [complement_naive_bayes/](./complement_naive_bayes/README.md) | Complement naive Bayes parameters. |
| [categorical_naive_bayes/](./categorical_naive_bayes/README.md) | Categorical naive Bayes parameters. |
| [extra_tree/](./extra_tree/README.md) | Single extremely randomized tree parameters. |
| [gaussian_process/](./gaussian_process/README.md) | Gaussian process parameters. |
| [nearest_centroid/](./nearest_centroid/README.md) | Nearest centroid parameters. |
| [nu_svc/](./nu_svc/README.md) | Nu support vector classification parameters, kernels, and gamma rules. |
| [perceptron/](./perceptron/README.md) | Perceptron parameters and penalties. |
| [radius_neighbors/](./radius_neighbors/README.md) | Radius neighbors parameters, outlier handling, and related enums. |
| [ridge_classifier/](./ridge_classifier/README.md) | Ridge classifier parameters. |
| [bagging/](./bagging/README.md) | Bagging parameters. |
| [stacking/](./stacking/README.md) | Stacking parameters. |
| [voting/](./voting/README.md) | Voting parameters. |
| [baseline/](./baseline/README.md) | Feature-independent reference classifier. |
