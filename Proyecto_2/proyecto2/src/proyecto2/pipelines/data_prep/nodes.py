"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""
import pandas as pd
import os
import numpy as np
import random
from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import ( 
    OneHotEncoder, 
    OrdinalEncoder, 
    StandardScaler, 
    MinMaxScaler, 
    PowerTransformer, 
    FunctionTransformer, 
    RobustScaler
)

def get_data():
    raw_bank_data = pd.read_parquet(os.path.join("../data", "dataset.pq"))
    return raw_bank_data

def replace_null_values(raw_bank_data: pd.DataFrame,
                        params: Dict,                        
                        ):
    

    # Primero, se extraen los parámetros:

    # Lambdas de distribuciones de Poisson:
    lambdas_Poiss_Dists = params["lambdas_Poiss_Dists"]

    lambda_mis_cs0 = lambdas_Poiss_Dists["MonthInSal_cs0"]
    lambda_mis_cs1 = lambdas_Poiss_Dists["MonthInSal_cs1"]
    lambda_mb_cs0 = lambdas_Poiss_Dists["MonthBalance_cs0"]
    lambda_mb_cs1 = lambdas_Poiss_Dists["MonthBalance_cs1"]

    # Posibles valores y probabilidades de distribuciones discretas
    Probs_NDelayP_CS0 = params["discrete_probs_NDelayP_CS0"]
    PValues_NDelayP_cs0 = list(Probs_NDelayP_CS0.keys())
    Probs_NDelayP_cs0 = list(Probs_NDelayP_CS0.values())

    Probs_NDelayP_CS1 = params["discrete_probs_NDelayP_CS1"]
    PValues_NDelayP_cs1 = list(Probs_NDelayP_CS1.keys())
    Probs_NDelayP_cs1 = list(Probs_NDelayP_CS1.values())

    Probs_CredInq_CS0 = params["discrete_probs_CredInq_CS0"]
    PValues_CInq_cs0 = list(Probs_CredInq_CS0.keys())
    Probs_CInq_cs0 = list(Probs_CredInq_CS0.values())

    Probs_CredInq_CS1 = params["discrete_probs_CredInq_CS1"]
    PValues_CInq_cs1 = list(Probs_CredInq_CS1.keys())
    Probs_CInq_cs1 = list(Probs_CredInq_CS1.values())

    Probs_CredHistA_CS0 = params["discrete_probs_CredHistA_CS0"]
    PValues_CHistA_cs0 = list(Probs_CredHistA_CS0.keys())
    Probs_CHistA_cs0 = list(Probs_CredHistA_CS0.values())

    Probs_CredHistA_CS1 = params["discrete_probs_CredHistA_CS1"]
    PValues_CHistA_cs1 = list(Probs_CredHistA_CS1.keys())
    Probs_CHistA_cs1 = list(Probs_CredHistA_CS1.values())

    # Medias y Desviaciones estándar de distribuciopnes normales:
    MS_Normal_Dists = params["MeansStds_Normal_Dists"]
    Mean_CCredL_cs0 = MS_Normal_Dists["Mean_ChangCredLim_cs0"]
    Mean_CCredL_cs1 = MS_Normal_Dists["Mean_ChangCredLim_cs1"]
    Std_CCredL_cs0 = MS_Normal_Dists["Std_ChangCredLim_cs0"]
    Std_CCredL_cs1 = MS_Normal_Dists["Std_ChangCredLim_cs1"]

    # Probabilidades Distribuciones Geométricas:
    Probs_Geom_Dists = params["Probs_Geom_Dists"]
    PGeom_AInv_cs0 = Probs_Geom_Dists["PGeom_AmInvMonthly_cs0"]
    PGeom_AInv_cs1 = Probs_Geom_Dists["PGeom_AmInvMonthly_cs1"]

    # Ahora, se carga el dataframe y se 
    # cambia el valor en monthly_balance que es menor a 0 por NaN:
    df = raw_bank_data
    cond = df['monthly_balance'] < 0
    df.loc[cond, 'monthly_balance'] = pd.NA


    # Ahora se reemplazan los valores nulos:

    # monthly_inhand_salary:
    cond0 = (df['credit_score'] == 0) & (df['monthly_inhand_salary'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['monthly_inhand_salary'].isnull())

    newCS0_MInS_values = np.random.poisson(lambda_mis_cs0, size=cond0.sum())
    newCS1_MInS_values = np.random.poisson(lambda_mis_cs1, size=cond1.sum())
    
    df.loc[cond0, 'monthly_inhand_salary'] = newCS0_MInS_values[:sum(cond0)]
    df.loc[cond1, 'monthly_inhand_salary'] = newCS1_MInS_values[:sum(cond1)]

    # monthly_balance:
    cond0 = (df['credit_score'] == 0) & (df['monthly_balance'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['monthly_balance'].isnull())

    newCS0_MB_values = np.random.poisson(lambda_mb_cs0, size=sum(cond0))
    newCS1_MB_values = np.random.poisson(lambda_mb_cs1, size=sum(cond0))
    
    df.loc[cond0, 'monthly_balance'] = newCS0_MB_values[:sum(cond0)]
    df.loc[cond1, 'monthly_balance'] = newCS1_MB_values[:sum(cond1)]
    
    # num_of_delayed_payment
    cond0 = (df['credit_score'] == 0) & (df['num_of_delayed_payment'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['num_of_delayed_payment'].isnull())
    newCS0_NDP_values = random.choices(PValues_NDelayP_cs0, 
                                       weights=Probs_NDelayP_cs0, k=sum(cond0))
    newCS1_NDP_values = random.choices(PValues_NDelayP_cs1, 
                                       weights=Probs_NDelayP_cs1, k=sum(cond1))
    df.loc[cond0, 'num_of_delayed_payment'] = newCS0_NDP_values[:sum(cond0)]
    df.loc[cond1, 'num_of_delayed_payment'] = newCS1_NDP_values[:sum(cond1)]

    # num_credit_inquiries
    cond0 = (df['credit_score'] == 0) & (df['num_credit_inquiries'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['num_credit_inquiries'].isnull())
    newCS0_CInq_values = random.choices(PValues_CInq_cs0, 
                                       weights=Probs_CInq_cs0, k=sum(cond0))
    newCS1_CInq_values = random.choices(PValues_CInq_cs1, 
                                       weights=Probs_CInq_cs1, k=sum(cond0))
    df.loc[cond0, 'num_credit_inquiries'] = newCS0_CInq_values[:sum(cond0)]
    df.loc[cond1, 'num_credit_inquiries'] = newCS1_CInq_values[:sum(cond1)]

    # credit_history_age
    cond0 = (df['credit_score'] == 0) & (df['credit_history_age'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['credit_history_age'].isnull())
    newCS0_CHA_values = random.choices(PValues_CHistA_cs0, 
                                       weights=Probs_CHistA_cs0, k=sum(cond0))    
    newCS1_CHA_values = random.choices(PValues_CHistA_cs1, 
                                       weights=Probs_CHistA_cs1, k=sum(cond0))
    df.loc[cond0, 'credit_history_age'] = newCS0_CHA_values[:sum(cond0)]
    df.loc[cond1, 'credit_history_age'] = newCS1_CHA_values[:sum(cond1)]

    # changed_credit_limit
    cond0 = (df['credit_score'] == 0) & (df['changed_credit_limit'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['changed_credit_limit'].isnull())
    newCS0_CCL_values = np.random.normal(loc=Mean_CCredL_cs0, 
                                         scale=Std_CCredL_cs0,
                                         size=sum(cond0))   
    newCS1_CCL_values = np.random.normal(loc=Mean_CCredL_cs1, 
                                         scale=Std_CCredL_cs1,
                                         size=sum(cond1))
    df.loc[cond0, 'changed_credit_limit'] = newCS0_CCL_values[:sum(cond0)]
    df.loc[cond1, 'changed_credit_limit'] = newCS1_CCL_values[:sum(cond1)]

    # amount_invested_monthly
    cond0 = (df['credit_score'] == 0) & (df['amount_invested_monthly'].isnull())
    cond1 = (df['credit_score'] == 1) & (df['amount_invested_monthly'].isnull())
    newCS0_AIM_values = np.random.geometric(p=PGeom_AInv_cs0, size=sum(cond0))   
    newCS1_AIM_values = np.random.geometric(p=PGeom_AInv_cs1, size=sum(cond1)) 
    df.loc[cond0, 'amount_invested_monthly'] = newCS0_AIM_values[:sum(cond0)]
    df.loc[cond1, 'amount_invested_monthly'] = newCS1_AIM_values[:sum(cond1)]

    # Ahora verificamos que no se tengan valores nulos:
    assert df.isnull().sum().sum() == 0
    bank_data_no_nulls = df
    return bank_data_no_nulls
    

def get_column_transformer():
    num_vars_extreme_outliers = [
        "age",
        "annual_income",
        "num_bank_accounts",
        "num_credit_card",
        "interest_rate",
        "num_of_loan",
        "num_of_delayed_payment",
        "num_credit_inquiries",
        "total_emi_per_month",
        "amount_invested_monthly",
        "monthly_balance"        
    ]
    num_vars = [
        "monthly_inhand_salary",
        "delay_from_due_date",
        "changed_credit_limit",
        "outstanding_debt",
        "credit_utilization_ratio",
        "credit_history_age"
    ]
    categorical_vars = [
        "occupation",
        "payment_of_min_amount",
        "payment_behaviour"
    ]
    pass_through_vars = [
        "customer_id",       
        #"credit_score"
    ]

    Col_Transformer = ColumnTransformer(
        transformers = [
            #('passthrough', 'passthrough', pass_through_vars),
            ('OneHotEncoder', OneHotEncoder(sparse_output=False),  categorical_vars),
            ('RobustNumericalTransform', RobustScaler(), num_vars_extreme_outliers),
            ('RegularNumericalTransform', StandardScaler(), num_vars),
            ]
        )
    Col_Transformer = Col_Transformer.set_output(transform="pandas")

    return Col_Transformer


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train_No_Transforms, y_train = X[:train_idx], y[:train_idx]
    X_valid_No_Transforms, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test_No_Transforms, y_test = X[valid_idx:], y[valid_idx:]

    return (X_train_No_Transforms, 
            X_valid_No_Transforms, 
            X_test_No_Transforms, 
            y_train, 
            y_valid, 
            y_test)