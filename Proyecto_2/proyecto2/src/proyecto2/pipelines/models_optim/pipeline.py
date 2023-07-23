"""
This is a boilerplate pipeline 'models_optim'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    optim_best_models,
    evaluate_pipe
    
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=optim_best_models,
                inputs=["Col_Transformer",
                        "X_train",
                        "y_train",
                        "X_valid",
                        "y_valid"],
                outputs=["XGB_Opt",
                         "LGBM_Opt",
                         "reportXGB", 
                         "reportLGBM"],
                name="optim_models",
            ), 

            node(
                func=evaluate_pipe,
                inputs=["X_test",
                        "y_test",
                        "XGB_Opt"],
                outputs="XGB_report_Test",
                name="eval_best_model",
            ), 


            
        ]
    )
