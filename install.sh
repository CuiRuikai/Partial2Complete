#!/usr/bin/env sh

# Usage:
#   bash install.sh          - Install packages with --user flag (in user's home directory)
#   bash install.sh --global - Install packages globally (may require root privileges, depending on the python interpreter used)

HOME=`pwd`

# Parse command-line arguments
USE_USER_FLAG="--user"
if [ "$1" = "--global" ]; then
    USE_USER_FLAG=""
    echo "Installing globally."
else
    echo "Installing with --user flag in user's home directory."
fi

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
python setup.py install $USE_USER_FLAG

# PointOps
cd $HOME/extensions/pointops
python setup.py install $USE_USER_FLAG
