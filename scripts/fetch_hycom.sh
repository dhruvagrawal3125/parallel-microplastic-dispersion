#!/bin/sh
# Downloads daily HYCOM GLBy0.08 surface currents for one region through the
# public NetCDF Subset Service, one request per day (the 12:00 UTC slice), then
# converts them with prep_currents.py into data/<region>_hycom.nc. Days the
# server cannot deliver are skipped and reported; rerunning fetches only what
# is missing.
#
#   scripts/fetch_hycom.sh gulf_stream             # 60 days from 2024-01-01
#   scripts/fetch_hycom.sh agulhas 2024-01-01 30   # start date, number of days
set -eu

region=$1
start=${2:-2024-01-01}
days=${3:-60}
root=$(cd "$(dirname "$0")/.." && pwd)
raw="$root/data/raw/$region"
mkdir -p "$raw"

# west east south north, longitudes in HYCOM's 0..360 convention
case $region in
    npac_gyre)   box="200 230 25 40" ;;
    california)  box="225 245 25 45" ;;
    gulf_stream) box="285 315 30 45" ;;
    agulhas)     box="5 40 -45 -30" ;;
    *) echo "unknown region $region" >&2; exit 1 ;;
esac
set -- $box
base="https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/uv3z"

i=0
missing=""
while [ "$i" -lt "$days" ]; do
    day=$(python3 -c "from datetime import date, timedelta as d; print(date.fromisoformat('$start') + d(days=$i))")
    out="$raw/$day.nc"
    if [ ! -s "$out" ]; then
        printf 'fetching %s %s\n' "$region" "$day"
        # vertCoord=0 is the surface level.
        if curl -sS --fail --retry 2 --max-time 300 -o "$out.part" \
            "$base?var=water_u&var=water_v&west=$1&east=$2&south=$3&north=$4&horizStride=1&time=${day}T12:00:00Z&vertCoord=0&accept=netcdf4"; then
            mv "$out.part" "$out"
        else
            rm -f "$out.part"
            missing="$missing $day"
        fi
    fi
    i=$((i + 1))
done
[ -z "$missing" ] || echo "not available:$missing" >&2

python3 "$root/scripts/prep_currents.py" --region "$region" --lat-stride 2 \
    --out "$root/data/${region}_hycom.nc" "$raw"/*.nc
