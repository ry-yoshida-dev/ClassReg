from enum import Enum


class ClassificationMethod(Enum):
    """
    Classification method implemented by a processor.

    Attributes
    ----------
    LOGISTIC_REGRESSION
        Logistic regression.
    SVC
        Support vector classifier.
    K_NEAREST_NEIGHBORS
        K-nearest neighbors classifier.
    DECISION_TREE
        Decision tree classifier.
    RANDOM_FOREST
        Random forest classifier.
    HIST_GRADIENT_BOOSTING
        Histogram-based gradient boosting classifier.
    GAUSSIAN_NAIVE_BAYES
        Gaussian naive Bayes classifier.
    EXTRA_TREES
        Extremely randomized trees classifier.
    GRADIENT_BOOSTING
        Gradient boosting classifier.
    ADA_BOOST
        AdaBoost classifier.
    LINEAR_DISCRIMINANT_ANALYSIS
        Linear discriminant analysis.
    QUADRATIC_DISCRIMINANT_ANALYSIS
        Quadratic discriminant analysis.
    MULTILAYER_PERCEPTRON
        Multilayer perceptron classifier.
    STOCHASTIC_GRADIENT_DESCENT
        Linear classifier trained by stochastic gradient descent.
    LINEAR_SVC
        Linear support vector classifier.
    BERNOULLI_NAIVE_BAYES
        Bernoulli naive Bayes classifier.
    MULTINOMIAL_NAIVE_BAYES
        Multinomial naive Bayes classifier.
    COMPLEMENT_NAIVE_BAYES
        Complement naive Bayes classifier.
    CATEGORICAL_NAIVE_BAYES
        Categorical naive Bayes classifier.
    EXTRA_TREE
        Single extremely randomized tree classifier.
    GAUSSIAN_PROCESS
        Gaussian process classifier.
    NEAREST_CENTROID
        Nearest centroid classifier.
    NU_SVC
        Nu support vector classifier.
    PERCEPTRON
        Perceptron classifier.
    RADIUS_NEIGHBORS
        Radius neighbors classifier.
    RIDGE_CLASSIFIER
        Ridge classifier.
    BAGGING
        Ensemble of copies fitted on random subsets.
    STACKING
        Ensemble whose final method learns from base predictions.
    VOTING
        Ensemble that averages the predictions of its methods.
    BASELINE
        Feature-independent reference classifier.
    """

    LOGISTIC_REGRESSION = "LogisticRegression"
    SVC = "SVC"
    K_NEAREST_NEIGHBORS = "KNearestNeighbors"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    HIST_GRADIENT_BOOSTING = "HistGradientBoosting"
    GAUSSIAN_NAIVE_BAYES = "GaussianNaiveBayes"
    EXTRA_TREES = "ExtraTrees"
    GRADIENT_BOOSTING = "GradientBoosting"
    ADA_BOOST = "AdaBoost"
    LINEAR_DISCRIMINANT_ANALYSIS = "LinearDiscriminantAnalysis"
    QUADRATIC_DISCRIMINANT_ANALYSIS = "QuadraticDiscriminantAnalysis"
    MULTILAYER_PERCEPTRON = "MultilayerPerceptron"
    STOCHASTIC_GRADIENT_DESCENT = "StochasticGradientDescent"
    LINEAR_SVC = "LinearSVC"
    BERNOULLI_NAIVE_BAYES = "BernoulliNaiveBayes"
    MULTINOMIAL_NAIVE_BAYES = "MultinomialNaiveBayes"
    COMPLEMENT_NAIVE_BAYES = "ComplementNaiveBayes"
    CATEGORICAL_NAIVE_BAYES = "CategoricalNaiveBayes"
    EXTRA_TREE = "ExtraTree"
    GAUSSIAN_PROCESS = "GaussianProcess"
    NEAREST_CENTROID = "NearestCentroid"
    NU_SVC = "NuSVC"
    PERCEPTRON = "Perceptron"
    RADIUS_NEIGHBORS = "RadiusNeighbors"
    RIDGE_CLASSIFIER = "RidgeClassifier"
    BAGGING = "Bagging"
    STACKING = "Stacking"
    VOTING = "Voting"
    BASELINE = "Baseline"
