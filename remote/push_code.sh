#!/bin/bash
# Push local code to the Colab VM (session "hips") and unpack it in /content/hips.
cd "$(dirname "$0")/.." && COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/claude-501/hips_code.tgz hips targets remote
colab upload -s hips /tmp/claude-501/hips_code.tgz /content/hips_code.tgz 2>&1 | grep -v "colab\]"
printf 'import subprocess\nprint(subprocess.run("cd /content/hips && tar xzf /content/hips_code.tgz && ls", shell=True, capture_output=True, text=True).stdout)\n' | bash "$(dirname "$0")/cx.sh"
