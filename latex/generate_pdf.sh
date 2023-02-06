#!/bin/bash

# generate latex file from mardwon
# usage:
#   ./generate_pdf.sh LinkPrediction.md
#
# NOTE: use just the file name, not the actual path
# (omit the ../)
docker run --rm --volume "$(dirname `pwd`):/data" --user `id -u`:`id -g` \
    pandoc/latex --pdf-engine=lualatex -f markdown -s $1 -o latex/tmp.tex
