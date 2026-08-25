import numpy as np

from sklearn.dummy import DummyClassifier

from src.classical.baseline import train_evaluate


def test_train_evaluate_returns_expected_metrics():
    X_train = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [2.0, 3.0],
        [3.0, 2.0],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    y_test = np.array([0, 1, 2, 3])

    model = DummyClassifier(strategy="prior")

    result = train_evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Test Model",
    )

    assert result["name"] == "Test Model"
    assert result["model"] is model
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_weighted"] <= 1.0
    assert 0.0 <= result["f1_fall"] <= 1.0
    assert result["confusion_matrix"].shape == (4, 4)
    assert result["y_pred"].shape == y_test.shape


def test_svm_classifier_can_fit_qwisense_features():
    from sklearn.svm import SVC

    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [0.1, 0.1],
        [1.1, 1.1],
        [2.1, 2.1],
        [3.1, 3.1],
    ])
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert set(predictions).issubset({0, 1, 2, 3})


def test_random_forest_classifier_can_fit_qwisense_features():
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [0.1, 0.1],
        [1.1, 1.1],
        [2.1, 2.1],
        [3.1, 3.1],
    ])
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert set(predictions).issubset({0, 1, 2, 3})


def test_train_evaluate_confusion_matrix_counts_predictions():
    from sklearn.dummy import DummyClassifier

    X_train = np.array([
        [0.0], [1.0], [2.0], [3.0],
        [0.1], [1.1], [2.1], [3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0], [1.0], [2.0], [3.0],
    ])
    y_test = np.array([0, 1, 2, 3])

    model = DummyClassifier(strategy="prior")

    result = train_evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Confusion Test",
    )

    cm = result["confusion_matrix"]

    assert cm.shape == (4, 4)
    assert cm.sum() == len(y_test)
    assert result["y_pred"].shape == y_test.shape
