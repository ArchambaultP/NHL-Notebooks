from ift6758.data import import_dataset

def load_entire_dataset():
    for season_start_year in [2016,2017, 2018, 2019, 2020]:
        import_dataset(season_start_year,"R") # import data for the regular season
        import_dataset(season_start_year, "P") # import playoff data
load_entire_dataset()
