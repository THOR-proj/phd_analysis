import CPOL_func as cf

for year in range(2001, 2016):
    try:
        cf.get_CPOL_tracks(year, rain=True, save_rain=True, dt=str(year))
    except: 
        print('No data for year {}'.format(str(year)))
cf.combine_tracks()
