# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical

# data = pd.read_csv("Kepler Objects of Interest (KOI).csv")
# data.info()
# data.drop(["koi_teq_err1","koi_teq_err2","kepler_name","koi_tce_delivname",
#            "dec","ra","kepoi_name","koi_tce_plnt_num","koi_pdisposition"],
#           inplace=True, axis=1, errors='ignore')


# mapping = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
# target = data['koi_disposition'].map(mapping)

# print("Unique target values (before drop):", target.unique())
# print("Number of NaNs in target:", target.isnull().sum())

# X = data.drop(columns=['koi_disposition'], errors='ignore')
# y = target

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
# print("y_train unique:", np.unique(y_train))
# print("NaNs in y_train:", y_train.isnull().sum())

# numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
# print("Numeric columns used:", numeric_cols)

# numeric_pipeline = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),  
#     ("scaler", StandardScaler())
# ])

# preprocessor = ColumnTransformer(transformers=[
#     ("num", numeric_pipeline, numeric_cols)
# ], remainder='drop') 

# X_train_trans = preprocessor.fit_transform(X_train)
# X_test_trans = preprocessor.transform(X_test)

# print(f"-------------{preprocessor.get_feature_names_out()}")


# y_train_cat = to_categorical(y_train, num_classes=3)
# y_test_cat = to_categorical(y_test, num_classes=3)



# # # 7) بناء الموديل DNN
# # model = Sequential([
# #     Dense(8, activation='tanh', input_shape=(X_train_trans.shape[1],)),
# #     Dense(128, activation='tanh'),
# #     Dense(64, activation='tanh'),
# #     Dense(32, activation='tanh'),
# #     Dense(8, activation='tanh'),


# #     Dense(3, activation='softmax')  # 3 classes
# # ])

# # from tensorflow.keras.callbacks import EarlyStopping

# # # Early stopping: يوقف التدريب لو الـ val_loss ما تحسنش 5 مرات متتالية
# # early_stop = EarlyStopping(monitor='val_loss', 
# #                            patience=5, 
# #                            restore_best_weights=True)

# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # # 8) تدريب الموديل
# # history = model.fit(X_train_trans, y_train_cat, 
# #                     validation_split=0.2, 
# #                     epochs=50, 
# #                     batch_size=32, 
# #                     verbose=1,
# #                     callbacks=[early_stop]
# # )

# # # 9) تقييم
# # y_pred = model.predict(X_test_trans)
# # y_pred_classes = np.argmax(y_pred, axis=1)

# # print("✅ DNN Accuracy:", accuracy_score(y_test, y_pred_classes))
# # print("\nClassification Report:\n", classification_report(y_test, y_pred_classes))

# # # Confusion Matrix
# # cm = confusion_matrix(y_test, y_pred_classes)
# # plt.figure(figsize=(6,4))
# # sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
# #             xticklabels=['False Positive','Candidate','Confirmed'],
# #             yticklabels=['False Positive','Candidate','Confirmed'])
# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")
# # plt.title("Confusion Matrix - DNN")
# # plt.show()

# # # 10) عرض منحنى الدقة والخسارة
# # plt.figure(figsize=(12,4))
# # plt.subplot(1,2,1)
# # plt.plot(history.history['accuracy'], label='Train Acc')
# # plt.plot(history.history['val_accuracy'], label='Val Acc')
# # plt.legend()
# # plt.title("Accuracy over Epochs")

# # plt.subplot(1,2,2)
# # plt.plot(history.history['loss'], label='Train Loss')
# # plt.plot(history.history['val_loss'], label='Val Loss')
# # plt.legend()
# # plt.title("Loss over Epochs")
# # plt.show()

# # model = XGBClassifier(
# #     n_estimators=500,
# #     learning_rate=0.01,
# #     max_depth=8,
# #     subsample=0.8,
# #     colsample_bytree=0.8,
# #     objective='multi:softmax',
# #     num_class=3,
# #     eval_metric='mlogloss',
# #     random_state=42,
# #     use_label_encoder=False
# # )




# # model.fit(X_train_trans, y_train)

# # y_pred = model.predict(X_test_trans)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # cm = confusion_matrix(y_test, y_pred)
# # plt.figure(figsize=(6,4))
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #             xticklabels=['False Positive','Candidate','Confirmed'],
# #             yticklabels=['False Positive','Candidate','Confirmed'])
# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")
# # plt.title("Confusion Matrix - XGBoost")
# # plt.show()


