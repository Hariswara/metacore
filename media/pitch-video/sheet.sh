#!/bin/bash
# usage: sheet.sh out.jpg in1.png in2.png ...
o=$1; shift; n=$#; args=(); for f in "$@"; do args+=(-i "$f"); done
cols=2; rows=$(( (n+1)/2 ))
ffmpeg -y -loglevel error "${args[@]}" -filter_complex "$(for i in $(seq 0 $((n-1))); do printf "[$i]scale=960:540[v$i];"; done; for i in $(seq 0 $((n-1))); do printf "[v$i]"; done; [ $((n%2)) -eq 1 ] && printf "" ; echo "xstack=inputs=$n:layout=$(for i in $(seq 0 $((n-1))); do printf "%s_%s|" $(( (i%2)*960 )) $(( (i/2)*540 )); done | sed 's/|$//' | sed 's/\([0-9]*\)_\([0-9]*\)/\1_\2/g'):fill=black")" -q:v 3 "$o"
