#!/bin/bash
cd "C:/Users/Administrator/WorkBuddy/2026-10-16-08-03-21/ZSPhotoApp" || exit 1
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
M27="qwen3.8-27b-abliterated:latest"
M9="qwen3.8-9b-uncensored:latest"
set -e
"$PY" gen.py prompts/matting.txt core/matting.py "$M27"
echo DONE_MATTING
"$PY" gen.py prompts/specs.txt core/photo_specs.py "$M27"
echo DONE_SPECS
"$PY" gen.py prompts/phototab.txt ui/photo_tab.py "$M27"
echo DONE_PHOTOTAB
"$PY" gen.py prompts/scantab.txt ui/scan_tab.py "$M27"
echo DONE_SCANTAB
"$PY" gen.py prompts/beautify.txt core/beautify.py "$M9"
echo DONE_BEAUTIFY
"$PY" gen.py prompts/scanner.txt core/scanner.py "$M9"
echo DONE_SCANNER
"$PY" gen.py prompts/main.txt main.py "$M9"
echo ALL_GENERATED
