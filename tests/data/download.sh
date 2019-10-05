#!/bin/bash

DOWNLOAD_DIR="$(dirname "$0")/downloads"
SOURCES_FILE="$(dirname "$0")/sources.txt"

rm -rf "$DOWNLOAD_DIR" && mkdir -p "$DOWNLOAD_DIR"
while read -r line
do
    if [[ "$line" =~ ^# ]]; then
        # ignore comments in the file
        continue
    fi

    if [[ "$line" =~ \.gz ]]; then
        cd "$DOWNLOAD_DIR" && curl "$line" -O && gunzip -f "${line##*/}"
    else
        cd "$DOWNLOAD_DIR" && curl "$line" -O
    fi
done < "$SOURCES_FILE"
