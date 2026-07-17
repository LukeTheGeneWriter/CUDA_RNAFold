#!/bin/sh
# Wrapper for help2man: derives the target program name and its
# associated --opt-include files from the .ggo source file passed as $1.

prog_dir=$(dirname "$1")
prog=$(basename "$1" .ggo)

exec help2man -N --help-option=--detailed-help \
  --opt-include=./include/ref_package.inc \
  --opt-include=./include/${prog}.inc \
  --opt-include=./include/ref_energy_par.inc \
  "./cmdlopt.sh ${prog_dir}/${prog}"
