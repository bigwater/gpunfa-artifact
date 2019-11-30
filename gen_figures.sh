echo "Generate all figures in the paper from raw data. "
echo "It may take around 20 minutes. "
cd raw_data/scripts
python gen_all_figures.py
cd -
echo "Finished, you may find the figures in raw_data/figures"

