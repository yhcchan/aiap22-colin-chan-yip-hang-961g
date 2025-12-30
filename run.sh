#!/bin/bash

# Change to project root
cd "$(dirname "$0")"

# Add 'src' to PYTHONPATH so Python knows where to find it
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run Python script
python -m src.main
