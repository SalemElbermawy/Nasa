import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


class DataHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):
        # Columns to drop
        drop_cols = [
            "loc_rowid","toi","tid","rastr","decstr","dec","ra","toi_created","rowupdate",
            "pl_tranmiderr1","pl_tranmiderr2","pl_tranmidlim","st_pmra","st_pmraerr1", "st_pmraerr2",
            "st_pmralim","st_pmdec","st_pmdecerr1","st_pmdecerr2","st_pmdeclim",
            "pl_eqterr1","pl_eqterr2","pl_eqtlim","pl_orbpererr2","pl_trandurherr2",
            "pl_trandeperr2","pl_radeerr2","st_tmagerr2","st_disterr2","st_tefferr2",
            "st_loggerr2","st_raderr2","st_rad","st_disterr1"
        ]
        self.data.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Map target
        mapping = {
   "FP": 0,
    "PC": 1,
    "APC": 1,
    "FA": 1,
    "CP": 2,
    "KP": 2
        }
        self.data["tfopwg_disp"] = self.data["tfopwg_disp"].map(mapping)

        X = self.data.drop(columns=['tfopwg_disp'], errors='ignore')
        y = self.data['tfopwg_disp']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocessing pipeline
        numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        numeric_pipeline = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_cols)
        ], remainder='drop')

        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        return X_train_trans, X_test_trans, y_train, y_test, preprocessor


class RFModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=1500,       
            max_depth=20,            
            max_features='log2',     
            min_samples_split=8,     
            min_samples_leaf=1,      
            bootstrap=True,         
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        return y_pred

    def predict(self, new_data):
        return self.model.predict(new_data)

    # === Graphs ===
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=[0,1,2],
                    yticklabels=[0,1,2])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Random Forest")
        plt.show()


if __name__ == "__main__":
    handler = DataHandler("TOI_2025.10.03_01.58.13.csv")
    X_train, X_test, y_train, y_test, preprocessor = handler.preprocess()

    rf = RFModel()
    rf.train(X_train, y_train)

    y_pred = rf.evaluate(X_test, y_test)
    rf.plot_confusion_matrix(X_test, y_test)

    new_sample = X_test[0].reshape(1, -1)
    print("Random Forest Prediction:", rf.predict(new_sample))


import joblib

joblib.dump(rf.model, "rf_model_2.pkl")

joblib.dump(preprocessor, "preprocessor_2.pkl")

print("✅ Models saved successfully!")
