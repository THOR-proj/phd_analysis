import CPOL_func as cf
import sys
import argparse

sys.path.insert(0, '/home/563/esh563/TINT')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--radar', '-r', type=int, default=63,
    help='radar domain to get tracks for')
parser.add_argument(
    '--year', '-y', type=int, default=63,
    help='radar domain to get tracks for')

args = parser.parse_args()

print('Getting radar {}, year {}'.format(args.radar, args.year))
cf.get_ACCESS_season(args.radar, args.year)
