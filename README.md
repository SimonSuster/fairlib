
# Debiasing for automated quality assessment

## Installation

This extension of *fairlib* requires Python3.7+ and [Pytorch](https://pytorch.org) 1.10 or higher. Dependencies of the core
modules are listed in `requirements.txt`. We recommend using a [venv](https://docs.python.org/3/library/venv.html) or [conda](https://www.anaconda.com/) environment for installation.

To set up a development environment, run the following commands to clone the repository and install
*fairlib*:

```bash
git clone https://github.com/SimonSuster/fairlib
git checkout develop
cd ~/fairlib
python setup.py develop
```

Note that you will also need to have the EvidenceGRADEr library installed in `EVIDENCEGRADER_DIR`:

```bash
git clone git@bitbucket.org:aimedtech/evidencegrader.git
export PYTHONPATH=$EVIDENCEGRADER_DIR
```

## Obtaining the data
### TrialstreamerRoB
Download the Area and Sex datasets from [here](https://drive.google.com/drive/folders/1-WSPIZDIgKKi30VtBqrhOpbYRIrl4Ota?usp=share_link) and unpack somewhere. We'll refer to the directory in which we unpacked `DATA_ROB`.

### EvidenceGRADEr
A request to access the dataset should be referred first to the Cochrane Col
laboration by emailing [support@cochrane.org](support@cochrane.org). When Cochrane permits (at its discretion) the use ofthe data by the third party, it will grant a license to use the Cochrane Database of Systematic Reviews, including a clause that confirms that Cochrane allows us to grant third party access to the data set created in this work. After that, please get in touch with us at the e-mail address [sim.suster@gmail.com](sim.suster@gmail.com) and we will send you the dataset.

## Training and evaluating the models

### TrialstreamerRoB

Set the dataset path variable `data_dir` in `fairlib/tutorial/RoB/rr_settings_sex.py` and `fairlib/tutorial/RoB/rr_settings_area.py` to `DATA_ROB`. Also set `serialization_dir` to a location of your choice.

To describe the Sex dataset and run the trivial baselines:

```bash
cd fairlib/data/src/RiskOfBias/
python desc_dataset.py -input_dir $DATA_ROB/rob_abstract_dataset_sex/ -protected_group sex
```

To do the same for Area, just replace `sex` with `area`.

To train the different models using Sex dataset:

- Vanilla:

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_vanilla_sex.py
```

- DownS

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_bt_sex.py "Downsampling"
```

- ReS

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_bt_sex.py "Resampling"
```

- Rew

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_bt_sex.py "Reweighting"
```

- Adv

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_adv_sex.py 
```

- DAdv

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_dadv_sex.py 
```

- FCL

```bash
cd tutorial/RoB
PYTHONHASHSEED=0 python3.8 rr_debias_fcl_sex.py 
```



### EvidenceGRADEr
TODO.


## License

This project is distributed under the terms of
the [APACHE LICENSE, VERSION 2.0](https://www.apache.org/licenses/LICENSE-2.0). The license applies to all files in
the [GitHub repository](http://github.com/HanXudong/fairlib) hosting this file.
