"""
This is a boilerplate pipeline 'baseline_classifier'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_and_fit_pipelines,
    evaluate_pipes,
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_and_fit_pipelines,
                inputs=["Col_Transformer",
                         "X_train",
                         "y_train"],
                outputs=["Dummy_Pipe_B", 
                         "LR_Pipe_B", 
                         "KN_Pipe_B",
                         "DT_Pipe_B", 
                         "SVC_Pipe_B", 
                         "RF_Pipe_B",
                         "LGBM_Pipe_B", 
                         "XGB_Pipe_B"],
                name="create_fit_pipelines",
            ), 

            node(
                func=evaluate_pipes,
                inputs=[ "X_valid",
                         "y_valid",
                         "Dummy_Pipe_B", 
                         "LR_Pipe_B", 
                         "KN_Pipe_B",
                         "DT_Pipe_B", 
                         "SVC_Pipe_B", 
                         "RF_Pipe_B",
                         "LGBM_Pipe_B", 
                         "XGB_Pipe_B"],
                outputs=["recalls_class1",
                         "Dummy_Pipe_B_Report", 
                         "LR_Pipe_B_Report",
                         "KNN_Pipe_B_Report",
                         "DT_Pipe_B_Report", 
                         "SVC_Pipe_B_Report", 
                         "RF_Pipe_B_Report",
                         "LGBM_Pipe_B_Report", 
                         "XGB_Pipe_B_Report"],
                name="eval_pipelines",
            ),          
            
        ]
    )
