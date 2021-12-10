#!/bin/bash

BASE_DIR=/g/data/rt52/era5/pressure-levels/reanalysis
SAVE_DIR=/g/data/w40/esh563/era5/pressure-levels/reanalysis

subset () {

    ncea -d latitude,-15.0,-9.0 -d longitude,128.0,134.0 \
    ${BASE_DIR}/${4}/${1}/${4}_era5_oper_pl_${1}${2}01-${1}${2}${3}.nc \
    ${SAVE_DIR}/${4}/${1}/${4}_era5_oper_pl_${1}${2}01-${1}${2}${3}.nc
}

check_leap_year () {
    echo '2000, 2004, 2008, 2012, 2016' | grep -q ${1}
}

for year in $(seq 1998 2016); do
    echo "Subsetting year ${year}."
    for var in u v z; do
        echo "Subsetting ${var}."

        echo "Subsetting January."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0101-${year}0131.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 01 31 ${var}
        fi

        echo "Subsetting February."
        if (check_leap_year ${year}); then
            FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0201-${year}0229.nc
            if test -f "$FILE"; then
                echo "${FILE} exists."
            else
                subset ${year} 02 29 ${var}
            fi
        else
            FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0201-${year}0228.nc
            if test -f "$FILE"; then
                echo "${FILE} exists."
            else
                subset ${year} 02 28 ${var}
            fi
        fi

        echo "Subsetting March."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0301-${year}0331.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 03 31 ${var}
        fi

        echo "Subsetting April."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0401-${year}0430.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 04 30 ${var}
        fi

        echo "Subsetting May."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}0501-${year}0531.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 05 31 ${var}
        fi

        echo "Subsetting October."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}1001-${year}1031.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 10 31 ${var}
        fi

        echo "Subsetting November."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}1101-${year}1130.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 11 30 ${var}
        fi

        echo "Subsetting December."
        FILE=${SAVE_DIR}/${var}/${year}/${var}_era5_oper_pl_${year}1201-${year}1231.nc
        if test -f "$FILE"; then
            echo "${FILE} exists."
        else
            subset ${year} 12 31 ${var}
        fi
    done
done
