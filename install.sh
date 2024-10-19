#!/usr/bin/env sh
HOME=`pwd`

# Check if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" | grep -q True; then
    echo "CUDA detected, forcing CUDA build for both extensions..."
    USE_CUDA=1
else
    echo "No CUDA detected, proceeding with standard build..."
    USE_CUDA=0
fi

# Chamfer Distance
cd $HOME/extensions/chamfer_dist
if [ "$USE_CUDA" = "1" ]; then
    FORCE_CUDA=1 python setup.py install --user
else
    python setup.py install --user
fi

# PointOps
cd $HOME/extensions/pointops
if [ "$USE_CUDA" = "1" ]; then
    FORCE_CUDA=1 python setup.py install --user
else
    python setup.py install --user
fi
