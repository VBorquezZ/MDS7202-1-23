"""
This is a boilerplate pipeline 'models_optim'
generated using Kedro 0.18.11
"""
import logging

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

log = logging.getLogger(__name__)

def optim_best_models(ColumnTransformer, X_train, y_train, X_val, y_val):

    XGB_Opt_pipe = Pipeline([
        ("preprocesamiento", ColumnTransformer),
        ("XGB_OPT", XGBClassifier())
    ], memory = ".")

    p_grid_XGB = [
        {
            "XGB_OPT__n_estimators": [100,
                                       150,
                                       200],
            "XGB_OPT__booster" : ["gbtree", 
                                  "gblinear", 
                                  "dart"],
        }
    ]
    
             
    LGBM_Opt_pipe = Pipeline([
        ("preprocesamiento", ColumnTransformer),
        ("LGBM_OPT", LGBMClassifier())
    ], memory = ".")

    p_grid_LGBM = [
        {
            "LGBM_OPT__boosting_type": ["dart", 
                                         "gbdt", 
                                         #"rf"
                                        ],
            "LGBM_OPT__num_leaves": [31,
                                     40,
                                     50,
                                     60,
                                     70
                                    ],
            "LGBM_OPT__n_estimators": [100,
                                       150,
                                       ],
                                           
        }
    ]
    
    scorer = make_scorer(recall_score, pos_label=1)

    # Crear el objeto GridSearchCV con la métrica recall de la clase '1'
    gsXGB = GridSearchCV(XGB_Opt_pipe, 
                        p_grid_XGB, 
                        scoring=scorer, 
                        n_jobs=-1)
    
    gsLGBM = GridSearchCV(LGBM_Opt_pipe, 
                          p_grid_LGBM,
                          scoring=scorer, 
                          n_jobs=-1)

    # Ajustar el GridSearchCV a tus datos de entrenamiento
    gsXGB = gsXGB.fit(X_train, y_train.values.ravel())
    gsLGBM = gsLGBM.fit(X_train, y_train.values.ravel())

    log.info("gsDT best score: " + str(gsXGB.best_score_))
    log.info("gsDT best params: " + str(gsXGB.best_params_))
    log.info("gsLGBM best score: " + str(gsLGBM.best_score_))
    log.info("gsLGBM best params: " + str(gsLGBM.best_params_))

    y_predxgb = gsXGB.predict(X_val)
    y_predlgbm = gsLGBM.predict(X_val)
    reportXGB = classification_report(y_val, y_predxgb, output_dict=True)
    reportLGBM = classification_report(y_val, y_predlgbm, output_dict=True)
    reportXGB = pd.DataFrame(reportXGB)
    reportLGBM = pd.DataFrame(reportLGBM)
    return gsXGB, gsLGBM, reportXGB, reportLGBM

def evaluate_pipe(features, labels, pipeline):
    y_pred = pipeline.predict(features)
    report = pd.DataFrame(classification_report(labels, 
                                                y_pred, 
                                                output_dict=True))
    return report