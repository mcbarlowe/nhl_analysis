import os
import sqlalchemy as sa
import pandas as pd
#import psycopg2
import math

from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



'''
def fetch_data():
    engine = sa.create_engine(os.environ["EW_CONNECT"])
    all_sits_sql = """
    select
        player
        ,player_upper
        ,api_id::integer
        ,birthday
        ,season_age
        ,position
        ,position_type
        ,shoots
        ,team
        ,season
        ,session
        ,gp
        ,toi
        ,toi_gp
        ,toi_perc
        ,g
        ,a1
        ,a2
        ,points
        ,isf
        ,iff
        ,icf
    from skater_std_sum_ev
    where session = 'R'
    """

    pp_sql = """
    select
        player
        ,player_upper
        ,api_id::integer
        ,birthday
        ,season_age
        ,position
        ,position_type
        ,shoots
        ,team
        ,season
        ,session
        ,gp
        ,toi
        ,toi_gp
        ,toi_perc
        ,g
        ,a1
        ,a2
        ,points
        ,isf
        ,iff
        ,icf
    from skater_std_sum_pp
    where session = 'R'
    """
    all_sits_df = pd.read_sql(all_sits_sql, engine)
    pp_df = pd.read_sql(pp_sql, engine)

    return all_sits_df, pp_df
'''


def cv_model(x, y, model, n_estimators: int, features: list = []):

    feature_array = x[features]
    estimator = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            (
                "regressor",
                BaggingRegressor(model, n_estimators=n_estimators, bootstrap=True),
            ),
        ]
    )
    estimator.fit(feature_array, y.values[:, 0])

    y_hat_values = estimator.predict(feature_array)
    r_squared = estimator.score(feature_array, y)
    mse = mean_squared_error(y, y_hat_values)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y, y_hat_values)

    print(f"Building model with features: {features}")
    print(
        (
            "Baseline linear model training set metrics:\n"
            f"R^2 for test set: {round(r_squared, 4)}\n"
            f"Mean Squared Error for training set: {round(mse, 4)}\n"
            f"Root Mean Squared Error for training set: {round(rmse, 4)}\n"
            f"Mean Absolute Error for training set: {round(mae, 4)}\n"
        )
    )

    return estimator


def oos_stats(x_test, y_test, features, model, model_name):
    standardized_df = x_test[features]
    mse = mean_squared_error(y_test, model.predict(standardized_df))
    rmse = math.sqrt(mean_squared_error(y_test, model.predict(standardized_df)))
    mae = mean_absolute_error(y_test, model.predict(standardized_df))
    r_sq = model.score(standardized_df, y_test)
    print(f"Building {model_name} with features: {features}")
    print(
        (
            f"{model_name} model Out of Sample metrics:\n"
            f"R^2: {round(r_sq, 4)}\n"
            f"Mean Squared Error: {round(mse, 4)}\n"
            f"Root Mean Squared Error: {round(rmse, 4)}\n"
            f"Average Mean Absolute Error for test set: {round(mae, 4)}"
        )
    )
