#!/bin/bash
# Download and build OMNeT++ and INET once, with only what this project needs.
# Safe to run again: finished steps are skipped and nothing is fetched twice.
#
#   bash sim/setup_omnet.sh
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${PLCA_HOME:-$HOME/plca}
OMNET=omnetpp-6.4.0
INET=inet-4.7.0
OMNET_URL=https://github.com/omnetpp/omnetpp/releases/download/$OMNET/$OMNET-linux-x86_64.tgz
INET_URL=https://github.com/inet-framework/inet/releases/download/v4.7.0/$INET-src.tgz

# Each compile job can use over 1 GB of RAM, so memory sets the limit, not cores.
ram_gb=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)
JOBS=$(( ram_gb / 2 > 1 ? ram_gb / 2 : 1 ))
[ "$JOBS" -gt "$(nproc)" ] && JOBS=$(nproc)
echo "compile jobs: $JOBS"

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
    # Command-line runs only: no GUI, 3D view or embedded Python.
    sed -i -E 's/^WITH_QTENV=.*/WITH_QTENV=no/;
               s/^WITH_OSG=.*/WITH_OSG=no/;
               s/^WITH_OSGEARTH=.*/WITH_OSGEARTH=no/;
               s/^WITH_PYTHON=.*/WITH_PYTHON=no/;
               s/^WITH_SCAVE_PYTHON_BINDINGS=.*/WITH_SCAVE_PYTHON_BINDINGS=no/' configure.user
    # configure insists on 400 MB of Python packages only if the IDE is present.
    if [ -d ide ]; then mv ide ide.unused; fi
    # shellcheck disable=SC1091
    source ./setenv -q
    ./configure
    make -j"$JOBS" MODE=release base  # base = no sample programs
    touch .plca_built
    cd "$ROOT"
fi

# shellcheck disable=SC1091
source "$ROOT/$OMNET/setenv" -q

if [ ! -f "$INET/.plca_built" ]; then
    cd "$INET"
    # shellcheck disable=SC1091
    source ./setenv -q
    # Wireless, routing and tutorial code is not used on a wired PLCA bus.
    # shellcheck disable=SC2046
    opp_featuretool disable -f $(tr -d '\r' < "$HERE/inet_features_off.txt")
    make makefiles
    make -j"$JOBS" MODE=release
    touch .plca_built
fi

du -sh "$ROOT"
echo SETUP_OK
