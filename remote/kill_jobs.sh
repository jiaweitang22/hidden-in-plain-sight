#!/bin/bash
# Stop running experiment scripts and their python children (not the build cache).
pkill -f "remote/exp_" ; pkill -f "hips.option_" ; sleep 1; echo killed
