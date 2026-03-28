from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans

def explain_model(best_model_name):
    explanations = {
        "LogisticRegression": "Good for simple linear relationships and fast performance.",
        "DecisionTree": "Handles non-linear patterns but may overfit.",
        "RandomForest": "Handles complex relationships and reduces overfitting.",
        "LinearRegression": "Works well when data has linear relationship.",
        "KMeans": "Used for grouping similar data points without labels."
    }

    return explanations.get(best_model_name, "Model explanation not available")

def get_feature_importance(model, X):
    try:
        if hasattr(model, "feature_importances_"):
            return dict(zip(X.columns, model.feature_importances_))
        else:
            return "Feature importance not available"
    except:
        return "Error in feature importance"
# Classification
def run_classification(X, y):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    results = []
    trained_models = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        results.append((name, acc))
        trained_models[name] = model

    best_model_name, best_score = max(results, key=lambda x: x[1])
    best_model = trained_models[best_model_name]

    explanation = explain_model(best_model_name)
    importance = get_feature_importance(best_model, X)

    return results, (best_model_name, best_score), explanation, importance


# Regression
def run_regression(X, y):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor()
    }

    results = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        results.append((name, r2))

    best_model = max(results, key=lambda x: x[1])

    return results, best_model


# Clustering
def run_clustering(X):
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(X)

    return {"clusters": list(labels[:10])}