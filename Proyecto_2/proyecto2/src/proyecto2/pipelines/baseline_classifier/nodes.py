"""
This is a boilerplate pipeline 'baseline_classifier'
generated using Kedro 0.18.11
"""

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def create_and_fit_pipelines(ColumnTransformer,
                             X_train,
                             y_train):

    models = {
        "Dummy": DummyClassifier(strategy="stratified"),
        "Logistic_Regression": LogisticRegression(),
        "K_Neighbors": KNeighborsClassifier(),
        "Decision_Tree": DecisionTreeClassifier(),
        "Support_Vector_C": SVC(),
        "Random_Forest": RandomForestClassifier(),
        "Light_GBM": LGBMClassifier(),
        "XGBoost" : XGBClassifier()
    }

    pipelines = []
    for model_name in models.keys():
        pipe = Pipeline([
            ("preprocesamiento", ColumnTransformer),
            (model_name, models[model_name])
        ]).fit(X_train, y_train.values.ravel())
        pipelines.append(pipe)

    (Dummy_Pipe_B, LR_Pipe_B, KN_Pipe_B, 
     DT_Pipe_B, SVC_Pipe_B, RF_Pipe_B, LGBM_Pipe_B, 
     XGB_Pipe_B) = pipelines
    
    return (Dummy_Pipe_B, LR_Pipe_B, KN_Pipe_B,
            DT_Pipe_B, SVC_Pipe_B, RF_Pipe_B, 
            LGBM_Pipe_B, XGB_Pipe_B)

