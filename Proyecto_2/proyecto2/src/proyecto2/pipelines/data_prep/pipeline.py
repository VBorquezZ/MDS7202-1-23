"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_data,
    get_column_transformer,
    replace_null_values
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
            )
        ]
    )