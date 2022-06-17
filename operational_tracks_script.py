import CPOL_func as cf
import sys
import argparse

sys.path.insert(0, '/home/563/esh563/TINT')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--year', '-y', type=int, default=2020,
    help='year to get tracks for')

parser.add_argument(
    '--month', '-m', type=int, default=12,
    help='month to get tracks for')

parser.add_argument(
    '--radar', '-r', type=int, default=63,
    help='radar to get tracks for')

args = parser.parse_args()

print('Getting year {}, month {}, radar {}.'.format(
    args.year, args.month, args.radar))
cf.get_oper_month(args.radar, year=args.year, month=args.month)
