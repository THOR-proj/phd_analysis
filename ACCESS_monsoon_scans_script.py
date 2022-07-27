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

args = parser.parse_args()

save_dir = '/g/data/w40/esh563/TINT_tracks/'
fig_dir = '/g/data/w40/esh563/TINT_figures/'

print('Getting year {}'.format(args.year))
cf.gen_ACCESS_verification_figures(
    save_dir, fig_dir, radar=args.radar,
    year=2020, exclusions=['simple_duration_cond'],
    suffix='_monsoon_alt', start_date=np.datetime64('2021-02-06'),
    end_date=np.datetime64('2021-02-13'))
