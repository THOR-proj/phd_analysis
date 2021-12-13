import CPOL_func as cf
import sys
import argparse

sys.path.insert(0, '/home/563/esh563/TINT')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--years', '-y', type=str, default='',
    help='years to get tracks for')

args = parser.parse_args()
if args.years == '':
    years = sorted(list(set(range(1998, 2016)) - {2007, 2008}))
else:
    years = args.years
    years = list(map(int, years.split(',')))

for year in years:
    print('Getting year {}'.format(year))
    cf.get_CPOL_season(year)
