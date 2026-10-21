#!/bin/bash
# render_stills.sh SCENE frame1 frame2 ...
sc=$1; shift
for fr in "$@"; do npx remotion still src/index.ts S$sc out/s${sc}_$fr.png --frame=$fr --log=error 2>&1 | grep -iE "error" | head -5; done
