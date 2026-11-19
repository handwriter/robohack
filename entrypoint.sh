#!/usr/bin/env bash
set -eu

if [ "$#" -ne 1 ] || [ "$1" != ping ]; then
    exit 64
fi

printf '%s\n' '{"protocol_version":"1.0","supported_tasks":[],"self_trained_rl":false,"preferred_speed_mps":null}'
