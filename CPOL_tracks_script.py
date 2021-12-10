import CPOL_func as cf
import sys

sys.path.insert(0, '/home/563/esh563/TINT')

for year in list(set(range(1998, 2016)) - {2007, 2008}):
    cf.get_CPOL_season(year)
