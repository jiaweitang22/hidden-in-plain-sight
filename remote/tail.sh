#!/bin/bash
# Show the tail of a VM log: remote/tail.sh <logname> [nchars]
printf 'import os\np="/content/logs/%s.log"\nprint(open(p).read()[-%s:] if os.path.exists(p) else "no log")\n' "$1" "${2:-1500}" | bash "$(dirname "$0")/cx.sh"
