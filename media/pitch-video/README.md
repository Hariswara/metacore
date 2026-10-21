# MetaCore pitch video

A 1:58 motion-graphics film for the evaluation panel, built in [Remotion](https://www.remotion.dev/) (React rendered to MP4).

## Pipeline

1. `audio/gen2.py` synthesises the narration phrase by phrase with [Kokoro-82M](https://github.com/thewh1teagle/kokoro-onnx) (Apache-2.0), voice `af_heart`.
2. `audio/layout.py` places the phrases on a timeline and writes `src/timeline.json` (scene lengths and phrase cue frames) plus `vo_full.wav`.
3. `audio/geo_build.py` turns the coastline data into SVG paths in `src/geo.json`.
4. `audio/audio.py` composes the score and sound effects in code and mixes them under the voice.
5. `npx remotion render src/index.ts Pitch out/silent.mp4 --muted`, then mux with the mix in ffmpeg (loudness-normalised to -16 LUFS).

Every scene animates from `timeline.json`, so changing the script only means regenerating the audio and re-rendering.

## Data and licences

- Island coastlines: Survey Department of Sri Lanka / OCHA ROAP via geoBoundaries, CC BY 3.0 IGO.
- Region and globe outlines: Natural Earth, public domain.
- Figures: CEB Jaffna generation ledger 2025 (`data/processed/ceb_generation_tidy.csv`).
- Fonts: Space Grotesk and JetBrains Mono, SIL OFL 1.1.
- Music and sound effects: synthesised for this film.
