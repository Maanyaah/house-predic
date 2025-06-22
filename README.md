import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

test_df = pd.read_csv("test.csv")  
test_ids = test_df["Id"]
test_df = test_df.drop(columns=["Id"])
numerical_cols = test_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = test_df.select_dtypes(include=["object"]).columns.tolist()
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])
test_processed = preprocessor.fit_transform(test_df)
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
final_columns = numerical_cols + cat_feature_names.tolist()
processed_test_df = pd.DataFrame(test_processed, columns=final_columns)
print(processed_test_df.head())
