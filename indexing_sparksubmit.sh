#!/usr/bin/env bash
mainpath = 'C:\\Users\\maxca\\Documents\\GitHub\\codingchallenge_indexing\\Data\\'

spark-submit \
    --master local \
    ./src/challenge_indexing.py $mainpath