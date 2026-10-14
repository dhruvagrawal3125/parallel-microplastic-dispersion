#!/bin/sh
# Downloads daily OSCAR 0.25-degree surface current files from NASA PO.DAAC and
# converts them into one global file for the simulator.
#
#   scripts/fetch_oscar.sh 2025-12-01 31           # 31 daily files, FINAL product
#   scripts/fetch_oscar.sh 2026-09-01 30 nrt       # near-real-time product
#
# PO.DAAC needs a free NASA Earthdata login (https://urs.earthdata.nasa.gov).
# curl reads it from ~/.netrc; run scripts/earthdata_login.sh once to save it.
set -eu

start=${1:?usage: fetch_oscar.sh START_DATE DAYS [final|nrt]}
days=${2:-31}
kind=${3:-final}
root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/data/oscar"
mkdir -p "$dest"

case $kind in
    final) coll=OSCAR_L4_OC_FINAL_V2.0 ;;
    nrt)   coll=OSCAR_L4_OC_NRT_V2.0 ;;
    *) echo "product must be final or nrt" >&2; exit 1 ;;
esac
base="https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/$coll"

if ! grep -qs "urs.earthdata.nasa.gov" "$HOME/.netrc"; then
    echo "No Earthdata login in ~/.netrc. Run scripts/earthdata_login.sh first." >&2
    exit 1
fi

jar=$(mktemp)
trap 'rm -f "$jar"' EXIT
i=0
files=""
while [ "$i" -lt "$days" ]; do
    stamp=$(python3 -c "from datetime import date, timedelta as d; print((date.fromisoformat('$start') + d(days=$i)).strftime('%Y%m%d'))")
    name="oscar_currents_${kind}_${stamp}.nc"
    if [ ! -s "$dest/$name" ]; then
        echo "fetching $name"
        curl -sS --fail --location --netrc --cookie "$jar" --cookie-jar "$jar" --retry 2 --max-time 600 \
            -o "$dest/$name.part" "$base/$name"
        mv "$dest/$name.part" "$dest/$name"
    fi
    files="$files $dest/$name"
    i=$((i + 1))
done

out="$root/data/oscar_global_${start}_${days}d.nc"
# shellcheck disable=SC2086
python3 "$root/scripts/prep_currents.py" --region global --out "$out" $files
echo "next: python3 scripts/world_drift.py --field $out"
