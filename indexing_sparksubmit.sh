#!/usr/bin/env bash
mainpath="/home/ubuntu/codingchallenge_indexing/"
spark-submit \
    --master local \
    /home/ubuntu/codingchallenge_indexing/src/challenge_indexing.py $mainpath
