import sys
sys.path.insert(0, '/home/563/esh563/TINT')
import CPOL_func as cf
import argparse

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

years_months = [
    [args.year, 10], [args.year, 11], [args.year, 12],
    [args.year+1, 1], [args.year+1, 2], [args.year+1, 3],
    [args.year+1, 4]]

print('Getting year {}'.format(args.year))
for i in range(len(years_months)):
    cf.gen_operational_verification_figures(
        save_dir, fig_dir, radar=args.radar,
        year=years_months[i][0], month=years_months[i][1])
