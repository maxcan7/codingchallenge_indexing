#!/usr/bin/env bash
mainpath = '/home/ubuntu/codingchallenge_indexing/'
spark-submit \
    --master local \
    ./src/challenge_indexing.py $mainpath
