import CPOL_func as cf
import sys

sys.path.insert(0, '/home/563/esh563/TINT')

for year in range(2005, 2006):
    cf.get_CPOL_season(year)
