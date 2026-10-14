#!/bin/sh
# Saves your NASA Earthdata login to ~/.netrc so curl can download from PO.DAAC.
# You type the username and password yourself at the prompts; the password is
# not shown and is written only to ~/.netrc (readable by you alone).
#
#   scripts/earthdata_login.sh
set -eu

netrc="${NETRC:-$HOME/.netrc}"
host=urs.earthdata.nasa.gov

printf 'Earthdata username: '
IFS= read -r user
printf 'Earthdata password (hidden): '
stty -echo 2>/dev/null || true
trap 'stty echo 2>/dev/null || true' EXIT
IFS= read -r pass
stty echo 2>/dev/null || true
printf '\n'
[ -n "$user" ] && [ -n "$pass" ] || { echo "username and password must not be empty" >&2; exit 1; }

# Quote the password so symbols and spaces survive; escape \ and " inside it.
esc=$(printf '%s' "$pass" | sed 's/\\/\\\\/g; s/"/\\"/g')

umask 077
tmp=$(mktemp "$netrc.XXXXXX")
[ -f "$netrc" ] && grep -v "$host" "$netrc" > "$tmp" || true   # replace any earlier entry
printf 'machine %s login %s password "%s"\n' "$host" "$user" "$esc" >> "$tmp"
mv "$tmp" "$netrc"
chmod 600 "$netrc"
echo "saved login for $host to $netrc"
