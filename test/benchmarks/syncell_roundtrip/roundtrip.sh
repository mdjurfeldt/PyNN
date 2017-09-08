#/bin/bash

if python build_and_write_network.py; then
  python read_and_rewrite_network.py
fi
