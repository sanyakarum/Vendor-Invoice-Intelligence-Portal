from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )

    param_grid ={
    "n_estimators":[100,200,300],
    "max_depth":[None, 4, 5, 6],
    "min_samples_split":[2,3,5],
    "min_samples_leaf":[1,2,5],
    "criterion":["gini", "entropy"]
    }

    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator = rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs= -1
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def evaluate_model(model, X_test, y_test, model_name):
    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

    

    