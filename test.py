import more_itertools

renamed_columns = [
    "position",
    "shoots",
    "next_season_age",
    "g",
    "a1",
    "a2",
    "points",
    "toi",
    "gp",
    "isf",
    "iff",
    "icf",
    "g_pp",
    "a1_pp",
    "a2_pp",
    "points_pp",
    "toi_pp",
    "gp_pp",
    "isf_pp",
    "iff_pp",
    "season_next",
    "pos_D",
    "pos_D/F",
    "pos_F",
    "toi_gp",
    "sh_percent",
    "sh_percent_pp",
    "avg_goals_season",
    "avg_sh_perc",
    "sh_perc_diff",
    "g_avg_past_2_seasons",
]

test_cols = [
    "position",
    "shoots",
    "next_season_age",
    "g",
    "a1",
    "a2",
    "points",
    "toi",
    "gp",
    "icf",
    "g_pp",
    "a1_pp",
    "a2_pp",
    "points_pp",
    "toi_pp",
    "gp_pp",
    "isf_pp",
    "iff_pp",
    "season_next",
    "pos_D",
    "pos_D/F",
    "pos_F",
    "toi_gp",
    "sh_percent",
    "sh_percent_pp",
    "avg_goals_season",
    "avg_sh_perc",
    "sh_perc_diff",
    "g_avg_past_2_seasons",
]

feature_combos = list(more_itertools.powerset(test_cols))

import pickle

with open("column_combos", "wb") as fp:  # Pickling
    pickle.dump(feature_combos, fp)
