
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# from xgboost import XGBClassifier, plot_importance
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping


# from sklearn.decomposition import PCA

# data=pd.read_csv("TOI_2025.10.03_01.58.13.csv")

# # print(data.info())


# data.drop([
# "loc_rowid","toi","tid","rastr","decstr","dec","ra","toi_created","rowupdate","pl_tranmiderr1",
# "pl_tranmiderr2","pl_tranmidlim","st_pmra","st_pmraerr1", "st_pmraerr2", "st_pmralim", "st_pmdec",
# "st_pmdecerr1","st_pmdecerr2","st_pmdeclim","pl_eqterr1","pl_eqterr2","pl_eqtlim",


# "pl_orbpererr2",
# "pl_trandurherr2",
# "pl_trandeperr2",
# "pl_radeerr2",
# "st_tmagerr2",
# "st_disterr2",
# "st_tefferr2",
# "st_loggerr2",
# "st_raderr2",
# "st_rad",
# "st_disterr1"

# ],axis="columns",inplace=True)

# print(f"/////////////////// {data.shape}")
# print(data.info())



# num_cols = data.select_dtypes(include=['float64','int64']).columns.tolist()

# corr_matrix = data[num_cols].corr()

# plt.figure(figsize=(14,10))
# sns.heatmap(corr_matrix, annot=True,fmt=".2f", cmap="coolwarm",)
# plt.title("Correlation Matrix of Numeric Features")
# plt.xticks(rotation=45, ha='right', fontsize=9)
# plt.yticks(rotation=0, fontsize=9)
# # plt.show()

# # remove the features which has strong relation with another one
# threshold = 0.75
# high_corr = []
# for i in range(len(corr_matrix.columns)):
#     for j in range(i):
#         if abs(corr_matrix.iloc[i,j]) > threshold:
#             high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i,j]))

# print("Highly correlated features (>|0.9|):")
# for f in high_corr:
#     print(f)
# #---------------------------------------------
# print(data.head(5))
# print(data["tfopwg_disp"].unique())

# mapping = {
#     "CP": 1,   # Confirmed Planet
#     "PC": 2,   # Planet Candidate
#     "FP": 0,   # False Positive
#     "APC": 2,  # Ambiguous Planet Candidate
#     "KP": 1 ,   # Known Planet
#     "FA":2
# }
# data["tfopwg_disp"] = data["tfopwg_disp"].map(mapping)


# X = data.drop(columns=['tfopwg_disp'], errors='ignore')
# y = data['tfopwg_disp']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42,
# )
# from sklearn.impute import KNNImputer

# numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
# numeric_pipeline = Pipeline(steps=[
#     ("imputer", KNNImputer(n_neighbors=5)),
#     ("scaler", StandardScaler())
# ])
# preprocessor = ColumnTransformer(transformers=[
#     ("num", numeric_pipeline, numeric_cols)
# ], remainder='drop')

# X_train_trans = preprocessor.fit_transform(X_train)
# X_test_trans = preprocessor.transform(X_test)

# num_classes = len(y.unique())
# y_train_cat = to_categorical(y_train, num_classes=num_classes)
# y_test_cat = to_categorical(y_test, num_classes=num_classes)

# # ===================== XGBoost =====================

# xgb_model = XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.01,
#     max_depth=16,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='multi:softmax',
#     num_class=num_classes,
#     eval_metric='mlogloss',
#     random_state=42,
#     use_label_encoder=False
# )

# xgb_model.fit(X_train_trans, y_train)


# predict=xgb_model.predict(X_test_trans)

# print(accuracy_score(y_test,predict))
