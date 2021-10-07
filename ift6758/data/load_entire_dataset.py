from ift6758.data import import_dataset, GameType


def load_entire_dataset():
    for season_start_year in [2016, 2017, 2018, 2019, 2020]:
        import_dataset(season_start_year, GameType.regular)  # import data for the regular season
        import_dataset(season_start_year, GameType.playoffs)  # import playoff data
