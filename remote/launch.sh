#!/bin/bash
# Launch a shell command detached on the VM: remote/launch.sh <logname> "<command>"
printf 'import subprocess\nsubprocess.Popen(%s, shell=True)\nprint("launched")\n' "$(python3 -c 'import json,sys;print(json.dumps("cd /content/hips && nohup bash -c "+json.dumps(sys.argv[2])+" > /content/logs/"+sys.argv[1]+".log 2>&1 &"))' "$1" "$2")" | bash "$(dirname "$0")/cx.sh"
