import sys
sys.path.insert(0, '/home/563/esh563/TINT')
import CPOL_func as cf
import argparse
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--year', '-y', type=int, default=2020,
    help='year to generate verification scans for')

parser.add_argument(
    '--radar', '-r', type=int, default=63,
    help='radar region to generate verification scans for')

parser.add_argument(
    '--month', '-m', type=int, default=1,
    help='month to generate verification scans for')

args = parser.parse_args()

save_dir = '/g/data/w40/esh563/TINT_tracks/ACCESS_radar_base/'
fig_dir = '/g/data/w40/esh563/TINT_figures/'

start_date = np.datetime64('{:04}-{:02}-01'.format(args.year, args.month))
if args.month == 12:
    end_date = np.datetime64('{}-01-01'.format(args.year+1))
else:
    end_date = np.datetime64('{}-{}-01'.format(args.year, args.month))

print('Getting year {}'.format(args.year))
cf.gen_operational_verification_figures(
    save_dir, fig_dir, radar=args.radar,
    year=args.year, exclusions=['simple_duration_cond'],
    suffix='_{}_{}'.format(args.year, args.month),
    start_date=start_date, end_date=end_date)
