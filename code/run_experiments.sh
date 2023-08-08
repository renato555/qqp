#!/bin/bash

validation_path="./data/valid_split.csv"
test_path="./data/test_split.csv"

out_folder="./outputs_small_early_stopping"
all_experiments_folder="./data/experiments_small"

for experiment in "$all_experiments_folder"/*
do
    echo -e "Running experiment: $(basename $experiment .csv)\n"

    experiment_out_folder="$out_folder/$(basename $experiment .csv)"
    mkdir -p "$experiment_out_folder"

    for i in {1..5}
    do
        echo -e "Iteration: $i\n"
        python train.py "$experiment" "$validation_path" "$test_path" > "$experiment_out_folder/$i.out"
    done

done
