#!/bin/bash

BASE_DIR=/g/data/rt52/era5/pressure-levels/reanalysis
SAVE_DIR=/g/data/w40/esh563/era5/pressure-levels/reanalysis

subset () {

    ncea -d latitude,-15.0,-9.0 -d longitude,128.0,134.0 \
    ${BASE_DIR}/${4}/${1}/${4}_era5_oper_pl_${1}${2}01-${1}${2}${3}.nc \
    ${SAVE_DIR}/${4}/${1}/${4}_era5_oper_pl_${1}${2}01-${1}${2}${3}.nc
}

for year in $(seq 1998 2016); do
    echo "Subsetting year ${year}$. \n"
    for var in u v z; do
        echo "Subsetting ${var}$. \n"

        echo "Subsetting January. \n"
        subset ${year} 01 31 ${var}

        echo "Subsetting February. \n"
        echo '2000, 2004, 2008, 2012, 2016' | grep -q ${year}
        if [[$? -eq 0]; then
            subset ${year} 02 29 ${var}
        else
            subset ${year} 02 28 ${var}
        fi

        echo "Subsetting March. \n"
        subset ${year} 03 31 ${var}

        echo "Subsetting April."
        subset ${year} 04 31 ${var}

        echo "Subsetting October. \n"
        subset ${year} 10 31 ${var}

        echo "Subsetting November. \n"
        subset ${year} 11 30 ${var}

        echo "Subsetting December. \n"
        subset ${year} 12 31 ${var}
    done
done
