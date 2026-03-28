#!/usr/bin/env bash

function get_bin_directory {
    sed -n '/bin.directory=/p' "$1" | awk -F '=' '{print $2}'
}

function resolve_path {
    local target="$1"

    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$target"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$target"
        return
    fi

    if command -v perl >/dev/null 2>&1; then
        perl -MCwd=realpath -e 'print realpath($ARGV[0])' "$target"
        return
    fi

    if command -v realpath >/dev/null 2>&1; then
        realpath "$target"
        return
    fi

    if command -v readlink >/dev/null 2>&1; then
        readlink -f "$target" 2>/dev/null && return
    fi

    echo "$target"
}

USER_DIR=$PWD
INSTALL_DIR="${BASH_SOURCE[0]}"

while [ -h "$INSTALL_DIR" ]; do
  cd "$(dirname "$INSTALL_DIR")"
  INSTALL_DIR="$(readlink "$(basename "$INSTALL_DIR")")"
done

cd "$(dirname "$INSTALL_DIR")"

BIN_DIR=$(get_bin_directory ./interproscan.properties)

if [ -n "$INTERPROSCAN_CONF" ] && [ -f "$INTERPROSCAN_CONF" ]; then
    PROPERTIES="$(resolve_path "$INTERPROSCAN_CONF")"
    PROPERTY="-Dsystem.interproscan.properties=${PROPERTIES}"
    BIN_DIR="$(get_bin_directory "${PROPERTIES}")"
else
    PROPERTIES="./interproscan.properties"
    PROPERTY=""
fi

# Set environment variables for getorf
export EMBOSS_ACDROOT="$BIN_DIR"/nucleotide
export EMBOSS_DATA="$BIN_DIR"/nucleotide

# Check Java is installed
JAVA=$(type -p java)
if [ -z "$JAVA" ]; then
    echo "Java not found. Please install Java 11 and place it on your path,"
    echo "or edit the interproscan.sh script to refer to your Java installation."
    exit 1
fi

# Check Java version is supported
JAVA_VERSION=$("$JAVA" -Xms32M -Xmx32M -version 2>&1 | sed -n '/version/p' | awk -F '"' '{print $2}' )
JAVA_MAJOR_VERSION_FULL="$( cut -d ';' -f 1 <<< "$JAVA_VERSION" )"
JAVA_MAJOR_VERSION="${JAVA_MAJOR_VERSION_FULL%%.*}"
if [ "${JAVA_MAJOR_VERSION}" -lt 11 ];
then
    echo "Java version 11 is required to run InterProScan."
    echo "Detected version ${JAVA_VERSION}"
    echo "Please install the correct version."
    exit 1
fi

"$JAVA" \
 -XX:ParallelGCThreads=8 \
 -Xms2028M -Xmx14G \
 $PROPERTY \
 -jar interproscan-5.jar $@ -u $USER_DIR
