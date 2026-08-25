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


def test_plot_comparison_returns_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from src.classical.baseline import plot_comparison

    results = [
        {
            "name": "SVM",
            "accuracy": 0.85,
            "f1_fall": 0.80,
        },
        {
            "name": "Random Forest",
            "accuracy": 0.90,
            "f1_fall": 0.88,
        },
    ]

    output = tmp_path / "comparison.png"

    fig = plot_comparison(
        results,
        save_path=str(output),
    )

    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_confusion_matrix_saves_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from src.classical.baseline import plot_confusion_matrix

    cm = np.array([
        [5, 1, 0, 0],
        [0, 4, 1, 0],
        [0, 0, 5, 1],
        [0, 0, 1, 4],
    ])

    output = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(
        cm,
        title="Test Confusion Matrix",
        save_path=str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close("all")


def test_plot_comparison_handles_empty_results(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from src.classical.baseline import plot_comparison

    output = tmp_path / "empty_comparison.png"

    fig = plot_comparison(
        [],
        save_path=str(output),
    )

    assert fig is not None
    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_comparison_supports_multiple_models(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from src.classical.baseline import plot_comparison

    results = [
        {"name": "SVM", "accuracy": 0.80, "f1_fall": 0.75},
        {"name": "Random Forest", "accuracy": 0.90, "f1_fall": 0.85},
        {"name": "Extra Trees", "accuracy": 0.92, "f1_fall": 0.88},
    ]

    output = tmp_path / "multi_model_comparison.png"

    fig = plot_comparison(results, save_path=str(output))

    assert fig is not None
    assert len(fig.axes) == 1
    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_comparison_supports_single_model(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    from src.classical.baseline import plot_comparison

    results = [
        {
            "name": "SVM",
            "accuracy": 0.85,
            "f1_fall": 0.80,
        },
    ]

    output = tmp_path / "single_model_comparison.png"

    fig = plot_comparison(results, save_path=str(output))

    assert fig is not None
    assert len(fig.axes) == 1
    assert output.exists()
    assert output.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_train_evaluate_metrics_are_bounded():
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

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Metric Bounds",
    )

    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_weighted"] <= 1.0
    assert 0.0 <= result["f1_fall"] <= 1.0


def test_train_evaluate_predictions_match_test_size():
    from sklearn.dummy import DummyClassifier

    X_train = np.array([
        [0.0], [1.0], [2.0], [3.0],
        [0.1], [1.1], [2.1], [3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0], [1.0], [2.0], [3.0],
        [0.5], [1.5],
    ])
    y_test = np.array([0, 1, 2, 3, 0, 1])

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Prediction Size",
    )

    assert len(result["y_pred"]) == len(y_test)
    assert result["y_pred"].ndim == 1


def test_svm_probability_output_has_expected_shape():
    from sklearn.svm import SVC

    X_train = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
        [0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_train[:4])

    assert probabilities.shape == (4, 4)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(4),
        rtol=1e-6,
        atol=1e-6,
    )


def test_random_forest_is_reproducible_with_fixed_seed():
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
        [0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1],
    ])
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    model_1 = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )
    model_2 = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model_1.fit(X, y)
    model_2.fit(X, y)

    predictions_1 = model_1.predict(X)
    predictions_2 = model_2.predict(X)

    np.testing.assert_array_equal(predictions_1, predictions_2)


def test_random_forest_uses_expected_configuration():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    assert model.n_estimators == 100
    assert model.random_state == 42


def test_svm_uses_expected_configuration():
    from sklearn.svm import SVC

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    assert model.kernel == "rbf"
    assert model.C == 10
    assert model.gamma == "scale"
    assert model.probability is True
    assert model.random_state == 42


def test_random_forest_uses_expected_configuration():
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    assert model.n_estimators == 100
    assert model.random_state == 42


def test_svm_uses_expected_configuration():
    from sklearn.svm import SVC

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42,
    )

    assert model.kernel == "rbf"
    assert model.C == 10
    assert model.gamma == "scale"
    assert model.probability is True
    assert model.random_state == 42


def test_train_evaluate_result_contains_all_expected_fields():
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

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Structure Test",
    )

    expected_keys = {
        "name",
        "model",
        "accuracy",
        "f1_weighted",
        "f1_fall",
        "confusion_matrix",
        "y_pred",
    }

    assert set(result.keys()) == expected_keys


def test_train_evaluate_predictions_use_known_labels():
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

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Label Test",
    )

    assert set(result["y_pred"]).issubset({0, 1, 2, 3})


def test_train_evaluate_with_real_svm():
    from sklearn.svm import SVC

    X_train = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
        [0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
    ])
    y_test = np.array([0, 1, 2, 3])

    result = train_evaluate(
        SVC(kernel="rbf", C=10, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
        "SVM Test",
    )

    assert result["name"] == "SVM Test"
    assert result["confusion_matrix"].shape == (4, 4)
    assert len(result["y_pred"]) == len(y_test)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_train_evaluate_with_real_random_forest():
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
        [0.1, 0.1], [1.1, 1.1], [2.1, 2.1], [3.1, 3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
    ])
    y_test = np.array([0, 1, 2, 3])

    result = train_evaluate(
        RandomForestClassifier(
            n_estimators=20,
            random_state=42,
        ),
        X_train,
        X_test,
        y_train,
        y_test,
        "Random Forest Test",
    )

    assert result["name"] == "Random Forest Test"
    assert result["confusion_matrix"].shape == (4, 4)
    assert len(result["y_pred"]) == len(y_test)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_train_evaluate_preserves_model_identity():
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
        "Identity Test",
    )

    assert result["model"] is model
    assert hasattr(result["model"], "classes_")


def test_train_evaluate_model_contains_all_training_classes():
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

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Classes Test",
    )

    np.testing.assert_array_equal(
        result["model"].classes_,
        np.array([0, 1, 2, 3]),
    )


def test_train_evaluate_preserves_test_sample_count():
    from sklearn.dummy import DummyClassifier

    X_train = np.array([
        [0.0], [1.0], [2.0], [3.0],
        [0.1], [1.1], [2.1], [3.1],
    ])
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    X_test = np.array([
        [0.0], [1.0], [2.0], [3.0],
        [0.5], [1.5],
    ])
    y_test = np.array([0, 1, 2, 3, 0, 1])

    result = train_evaluate(
        DummyClassifier(strategy="prior"),
        X_train,
        X_test,
        y_train,
        y_test,
        "Sample Count Test",
    )

    assert result["confusion_matrix"].sum() == len(y_test)
    assert len(result["y_pred"]) == len(y_test)
