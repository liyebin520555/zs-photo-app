#!/bin/bash
cd "C:/Users/Administrator/WorkBuddy/2026-10-16-08-03-21/ZSPhotoApp" || exit 1
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
M27="qwen3.8-27b-abliterated:latest"
M9="qwen3.8-9b-uncensored:latest"
set -e
"$PY" gen.py prompts/retouch.txt core/retouch.py "$M27"
echo DONE_RETOUCH
"$PY" gen.py prompts/phototab2.txt ui/photo_tab.py "$M27"
echo DONE_PHOTOTAB
"$PY" gen.py prompts/comfy.txt core/comfy_client.py "$M27"
echo DONE_COMFY
"$PY" gen.py prompts/layout.txt core/layout.py "$M9"
echo DONE_LAYOUT
"$PY" gen.py prompts/specdata.txt core/spec_data.py "$M9"
echo ALL_GENERATED_2
