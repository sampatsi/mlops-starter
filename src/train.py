
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from data import load_data


def main(args):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("iris-exp")

    X_train, X_test, y_train, y_test, _ = load_data()

    with mlflow.start_run() as run:
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth
        })
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"Run ID: {run.info.run_id}  F1: {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=3)
    args = parser.parse_args()
    main(args)
