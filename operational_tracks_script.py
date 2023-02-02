import CPOL_func as cf
import sys
import argparse
import os

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

parser.add_argument(
    '--tracks', '-t', type=str, default='/g/data/w40/esh563/TINT_tracks/',
    help='directory to save tracks')

args = parser.parse_args()

if not os.path.exists(args.tracks):
    os.makedirs(args.tracks)

print('Getting year {}, month {}, radar {}.'.format(
    args.year, args.month, args.radar))
cf.get_oper_month(
    args.radar, year=args.year, month=args.month, save_dir=args.tracks)
