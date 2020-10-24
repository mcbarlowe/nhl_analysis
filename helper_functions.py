import os
import sqlalchemy as sa
import psycopg2


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
    from skater_std_sum_all
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
