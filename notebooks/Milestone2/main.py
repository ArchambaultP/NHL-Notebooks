from ift6758.data import import_dataset
from ift6758.features import tidy_data as td

def main():
    seasons = [2015,2016,2017,2018]
    for s in seasons:
        raw_data  = import_dataset(s, "P", returnData=True)
        pbp_data = td.get_playbyplay_data(raw_data)
        pdp_tidied = td.tidy_playbyplay_data(pbp_data)
        print(pdp_tidied)
    return

if __name__ == "__main__":
    main()