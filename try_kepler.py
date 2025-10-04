import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


class DataHandler:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess(self):

        self.data.drop([
            "koi_teq_err1","koi_teq_err2","kepler_name",
            "koi_tce_delivname","dec","ra","kepoi_name",
            "koi_tce_plnt_num","koi_pdisposition"
        ], inplace=True, axis=1, errors='ignore')

        mapping = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
        target = self.data['koi_disposition'].map(mapping)

        X = self.data.drop(columns=['koi_disposition'], errors='ignore')
        y = target

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocessing pipeline
        numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, numeric_cols)
        ], remainder='drop')

        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        return X_train_trans, X_test_trans, y_train, y_test, preprocessor


class DNNModel:
    def __init__(self, input_dim, num_classes=3):
        self.model = Sequential([
            Dense(128, activation='tanh', input_shape=(input_dim,)),
                        Dropout(0.3),

            Dense(64, activation='tanh'),
                        Dropout(0.3),

            Dense(32, activation='tanh'),
                        Dropout(0.3),

            Dense(16, activation='tanh'),
                        Dropout(0.3),

            Dense(8, activation='tanh'),

            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.history = None

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        y_train_cat = to_categorical(y_train, num_classes=3)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        pred_classes = np.argmax(preds, axis=1)
        return("✅ DNN Accuracy:", accuracy_score(y_test, pred_classes))
        

    def predict(self, new_data):
        preds = self.model.predict(new_data)
        return np.argmax(preds, axis=1)

    # === Graphs ===
    def plot_confusion_matrix(self, X_test, y_test):
        preds = np.argmax(self.model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=['False Positive','Candidate','Confirmed'],
                    yticklabels=['False Positive','Candidate','Confirmed'])
        plt.title("Confusion Matrix - DNN")
        plt.show()

    def plot_learning_curves(self):
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.history.history['accuracy'], label='Train Acc')
        plt.plot(self.history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title("Accuracy over Epochs")

        plt.subplot(1,2,2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title("Loss over Epochs")
        plt.show()


class XGBModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            use_label_encoder=False
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return("✅ XGBoost Accuracy:", accuracy_score(y_test, y_pred))

    def predict(self, new_data):
        return self.model.predict(new_data)

    # === Graphs ===
    def plot_confusion_matrix(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=['False Positive','Candidate','Confirmed'],
                    yticklabels=['False Positive','Candidate','Confirmed'])
        plt.title("Confusion Matrix - XGBoost")
        plt.show()

    def plot_feature_importance(self):
        plot_importance(self.model, importance_type="gain")
        plt.title("Feature Importance - XGBoost")
        plt.show()


if __name__ == "__main__":
    handler = DataHandler("Kepler Objects of Interest (KOI).csv")
    X_train, X_test, y_train, y_test, preprocessor = handler.preprocess()

    dnn = DNNModel(input_dim=X_train.shape[1])
    dnn.train(X_train, y_train)
    dnn.evaluate(X_test, y_test)
    dnn.plot_confusion_matrix(X_test, y_test)
    dnn.plot_learning_curves()

    xgb = XGBModel()
    xgb.train(X_train, y_train)
    xgb.evaluate(X_test, y_test)
    xgb.plot_confusion_matrix(X_test, y_test)

    new_sample = X_test[0].reshape(1, -1)
    print("DNN Prediction:", dnn.predict(new_sample))
    print("XGB Prediction:", xgb.predict(new_sample))





import joblib

joblib.dump(xgb.model, "xgb_model.pkl")

dnn.model.save("dnn_model.h5")

joblib.dump(preprocessor, "preprocessor.pkl")
