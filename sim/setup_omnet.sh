#!/bin/bash
# Download and build OMNeT++ and INET once, with only what this project needs.
# Safe to run again: finished steps are skipped and nothing is fetched twice.
#
#   bash sim/setup_omnet.sh
set -euo pipefail

ROOT=${PLCA_HOME:-$HOME/plca}
OMNET=omnetpp-6.4.0
INET=inet-4.7.0
OMNET_URL=https://github.com/omnetpp/omnetpp/releases/download/$OMNET/$OMNET-linux-x86_64.tgz
INET_URL=https://github.com/inet-framework/inet/releases/download/v4.7.0/$INET-src.tgz

# Each compile job can use over 1 GB of RAM, so memory sets the limit, not cores.
ram_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
JOBS=$(( ram_gb / 2 > 1 ? ram_gb / 2 : 1 ))
[ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)

mkdir -p "$ROOT"
cd "$ROOT"

fetch() {  # resumes a partial file; a finished archive is never fetched again
    local name=$1 url=$2 dir=$3
    [ -d "$dir" ] && return
    curl -fL --retry 10 --retry-delay 15 --retry-connrefused -C - -o "$name" "$url"
    tar xzf "$name"
}

fetch "$OMNET-linux-x86_64.tgz" "$OMNET_URL" "$OMNET"
fetch "$INET-src.tgz" "$INET_URL" "$INET"

if [ ! -f "$OMNET/.plca_built" ]; then
    cd "$OMNET"
    # No GUI: runs are command-line only, and the GUI libraries are large.
    sed -i -E 's/^#?\s*WITH_QTENV=.*/WITH_QTENV=no/;
               s/^#?\s*WITH_OSG=.*/WITH_OSG=no/;
               s/^#?\s*WITH_OSGEARTH=.*/WITH_OSGEARTH=no/' configure.user
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    [ -f python/requirements.txt ] && .venv/bin/pip install --quiet -r python/requirements.txt
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # shellcheck disable=SC1091
    source ./setenv -q
    ./configure
    make -j"$JOBS" MODE=release
    touch .plca_built
    cd "$ROOT"
fi

if [ ! -f "$INET/.plca_built" ]; then
    # shellcheck disable=SC1091
    source "$OMNET/.venv/bin/activate"
    # shellcheck disable=SC1091
    source "$OMNET/setenv" -q
    cd "$INET"
    # shellcheck disable=SC1091
    source ./setenv -q
    # Wireless, routing and visualization code is not used on a wired PLCA bus.
    if [ -f "$ROOT/.plca_features" ]; then
        while read -r feature; do
            opp_featuretool disable -f "$feature" || true
        done < "$ROOT/.plca_features"
    fi
    make makefiles
    make -j"$JOBS" MODE=release
    touch .plca_built
fi

du -sh "$ROOT"
echo SETUP_OK
