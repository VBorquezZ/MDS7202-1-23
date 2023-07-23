"""
This is a boilerplate pipeline 'baseline_classifier'
generated using Kedro 0.18.11
"""
import logging

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report


log = logging.getLogger(__name__)

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
            (model_name+"_Baseline", models[model_name])
        ]).fit(X_train, y_train.values.ravel())
        pipelines.append(pipe)

    (Dummy_Pipe_B, LR_Pipe_B, KN_Pipe_B, 
     DT_Pipe_B, SVC_Pipe_B, RF_Pipe_B, LGBM_Pipe_B, 
     XGB_Pipe_B) = pipelines
    
    return (Dummy_Pipe_B, LR_Pipe_B, KN_Pipe_B,
            DT_Pipe_B, SVC_Pipe_B, RF_Pipe_B, 
            LGBM_Pipe_B, XGB_Pipe_B)

def evaluate_pipes(X_val, y_val, *pipelines):
    reports = []
    recalls_class1 = []

    for pipe in pipelines: 
        model_name = list(pipe.named_steps.keys())[1]
        
        y_pred = pipe.predict(X_val)
        
        report = classification_report(y_val, y_pred, output_dict=True)
        report_df = pd.DataFrame(report)
        #print(report_df)

        reports.append(report_df)
        recalls_class1.append((model_name, report['1']['recall']))

    recalls_class1 = sorted(recalls_class1, key=lambda x: x[1], reverse=True)
    recalls_class1 = pd.DataFrame(recalls_class1, columns=['Model_Name', 'Recall_C_1'])
    reports.insert(0, recalls_class1)

    
    return reports

 


   



