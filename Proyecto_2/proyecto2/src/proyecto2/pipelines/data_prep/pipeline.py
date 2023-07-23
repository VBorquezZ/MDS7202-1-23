"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_data,
    get_column_transformer,
    replace_null_values,
    split_data
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs="raw_bank_data",
                name="get_data_node",
            ),
            
            node(
                func=replace_null_values,
                inputs=["raw_bank_data",
                        "params:Params_Nulls_Replacement"],
                outputs="bank_data_no_nulls",
                name="process_null_data",
            ),

            node(
                func=split_data,
                inputs=["bank_data_no_nulls",
                        "params:split_params"],
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="split_data_node",
            ),

            node(
                func=get_column_transformer,
                inputs=None,
                outputs="Col_Transformer",
                name="get_col_transformer",
            ),
        ]
    )