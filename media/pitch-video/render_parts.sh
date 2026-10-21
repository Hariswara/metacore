#!/bin/bash
cd "$(dirname "$0")"
for s in 1 2 3 4 5 6 7 8 9 10 11 12; do
  st=$(date +%s)
  if npx remotion render out/bundle S$s out/parts/S$s.mp4 --concurrency=4 --codec=h264 --crf=16 --muted --log=error > out/parts/S$s.log 2>&1; then
    echo "scene $s done in $(( $(date +%s)-st ))s"
  else
    echo "scene $s FAILED"; tail -5 out/parts/S$s.log
  fi
done
echo ALLDONE
