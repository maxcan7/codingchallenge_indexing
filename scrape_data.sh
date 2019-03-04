#!/usr/bin/env bash
# Scrape the coding challenge repo for document files
for i in $(seq 1 44); 
do wget https://raw.githubusercontent.com/Samariya57/coding_challenges/master/data/indexing/$i; 
done
