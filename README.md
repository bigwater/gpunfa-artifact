# NFA Processing on GPU
## Artifact for *Why GPUs are Slow at Executing NFAs and How to Make them Faster* @ ASPLOS '2k20

*The artifact contains the source code and Python/shell scripts. The scripts generate the throughput results in Table 3 of our paper. We also provide our raw experimental data and the scripts to reproduce other figures in our paper.*

### Hardware dependencies
We developed and tested our work in two NVIDIA GPUs (Quadro P6000 and Tesla V100). We expect our code can run on GPUs with the compute capability no less than 5.0.

### Software dependencies
Our work requires CUDA 9.2 SDK. Our work uses cmake later than 3.10 for compilation. We use Python 3 for our scripts. 

#### Operating System
We test our work on Ubuntu 18.04. We expect it can work with mainstream Linux distributions.  

#### Required Python packages:

* matplotlib
* numpy
* pandas
* scipy
* xlrd

If you use `conda`, simply run
```
conda create --name gpunfa_env python=3.7 matplotlib numpy scipy pandas xlrd
conda activate gpunfa_env
```
to install all the required Python packages. 


### Datasets
All data sets are from public available benchmark suites. We convert the automata files of them to ANML format. To facilitate this step, we provide the data set that is ready to use. Our scripts will automatically unzip these datasets. 

### Benchmarks
If the benchmark could not be downloaded due to GitHub's quota, try to download it [here](https://www.dropbox.com/s/havbbf1281eer0i/gpunfa_benchmarks.zip?dl=0). 




### Installation
The `setup.sh` will automatically download the datasets, build the executables, and set up environmental variables. 

```
git clone https://github.com/bigwater/gpunfa-artifact.git 
cd gpunfa-artifact
source setup.sh
```

Our project contains three executables, `infant`, `ppopp12`, and `obat`. They are added to your PATH variable after the setup. The former two excutables are our implementations of iNFAnt and NFA-CG, respectively. The `obat` contains our schemes. They share the same options and arguments settings. 

### Basic Usage
```
obat -i [input_stream_file] -a [anml_file] -g [algorithm] ...
```
The algorithm could be: 

* obat2: the scheme NewTran
* obat_MC: NewTran + matchset compression
* hotstart_ea_no_MC2: Hotstart
* hotstart_ea: Howstart + matchset compression

```
ppopp12 -i [input_stream_file] -a [anml_file] -g [algorithm] ...
```
The algorithm could be:

* ppopp12: Our implementation of NFA-CG. 


```
infant -i [input_stream_file] -a [anml_file] -g [algorithm] ...
```

The algorithm could be:

* infant: Our implementation of iNFAnt. 


**Other options.** You can check how to use other options by showing the help using `obat -?` or `obat -h`. 


### Experiments
To get Table 3 in our paper, simply run the following commands.

```
cd gpunfa-artifact
./run_experiments_get_table3.sh
```

The entire set requires several hours to finish. It uses the same configuration as the paper and generates Table 3 automatically. It will generate a csv file `abs_throughput.csv` containing the information of Table 3. 


### Regenerating the figures of our paper from our raw data
```
cd ${GPUNFA_ROOT}
./gen_figures.sh 
```
The script will take around 20 min to finish. 


### Paper
Please refer to this paper for other details.  
```
@inproceedings{gpunfa-asplos2020,
 author = {Liu, Hongyuan and Pai, Sreepathi and Jog, Adwait},
 title = {{Why GPUs are Slow at Executing NFAs and How to Make them Faster}},
 booktitle = {Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
 year = {2020}
}
```



