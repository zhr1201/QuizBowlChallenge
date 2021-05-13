#!/usr/bin/env bash

if $(conda env list | grep -q "/opt/conda/envs/qb"); then
  echo "INFO: Using qb environment"
  source /opt/conda/etc/profile.d/conda.sh > /dev/null 2> /dev/null
  conda activate qb
else
  echo "INFO: Using base environemnt"
fi

python -m qanta.cli web --config-file conf/TFIDF-None.yaml
