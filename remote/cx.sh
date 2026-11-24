#!/bin/bash
# Robust "colab exec": reads Python from stdin, retries on dropped connections / timeouts.
code=$(cat)
for i in 1 2 3 4; do
  out=$(printf '%s\n' "$code" | timeout 150 colab exec -s hips 2>&1)
  if printf '%s' "$out" | grep -qE "Connection was lost|Traceback \(most recent call last\).*colab_cli|ReadTimeout" || [ -z "$out" ]; then
    sleep $((i * 5)); continue
  fi
  printf '%s\n' "$out" | grep -v "colab\]"; exit 0
done
echo "CX_FAILED"; exit 1
