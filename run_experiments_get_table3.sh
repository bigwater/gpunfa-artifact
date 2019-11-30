cd ${GPUNFA_ROOT}

if [ ! -d "exp_table3" ]; then
    mkdir exp_table3 && cd exp_table3
else
    cd exp_table3
fi

cp ../gpunfa_code/scripts/configs/* .

echo "Running Experiments... This will take several hours. "
python ../gpunfa_code/scripts/launch_exps.py -b app_spec -f exec_config_table3 -e --clean

echo "Experiments finished. "


if [ $? -eq 0 ]; then
    echo "Collecting experiment raw data."
    python ../gpunfa_code/scripts/collect_results.py -b app_spec -f exec_config_table3 

    echo "Generate the Table 3 from the raw data. "
    python ../gpunfa_code/scripts/ploting/abs_throughput_table.py 
    echo "You may find a csv file in the folder, which should be similar to Table 3 in our paper. "
else
    echo "Experiments terminate abnormally. "
    exit 1
fi

cd ${GPUNFA_ROOT}







