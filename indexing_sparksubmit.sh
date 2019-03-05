#!/usr/bin/env bash
mainpath = '/home/ubuntu/codingchallenge_indexing/data/'
spark-submit \
    --master local \
    ./src/challenge_indexing.py $mainpath
