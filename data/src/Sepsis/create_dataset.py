import argparse
import os
import statistics

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairlib.src.utils import PandasUtils, save_json

FNAME_DIAGNOSES = 'DIAGNOSES_ICD.csv.gz'
FNAME_NOTES = 'NOTEEVENTS.csv.gz'
FNAME_PATIENTS = 'PATIENTS.csv.gz'
FNAME_ADMISSIONS = 'ADMISSIONS.csv.gz'

# relevant ICD-9-CM codes 
ICD9_SEPSIS = "99591"
ICD9_SEVERE_SEPSIS = "99592"
ICD9_SEPTIC_SHOCK = "78552"

FNAME_LABELS = 'sepsis_labels.json'
FNAME_PROTECTED_LABELS = 'protected_labels.json'


def get_ethnicity(note):
    # ethnicity categories as defined in "https://aclanthology.org/2022.clinicalnlp-1.10.pdf": ‘WHITE’, ‘BLACK’, ‘ASIAN’, ‘HISPANIC’, ‘OTHER‘

    label = note["ETHNICITY"].item()
    if label.startswith("WHITE"):
        return "White"
    elif label.startswith("BLACK"):
        return "Black"
    elif label.startswith("ASIAN"):
        return "Asian"
    elif label.startswith("HISPANIC"):
        return "Hispanic"
    else:
        return "Other"


def get_sex(note):
    return note["GENDER"].item()


class SepsisMIMIC:
    # this sepsis extraction procedure is largely based on: https://github.com/clips/rnn_expl_rules/blob/master/src/datasets/sepsis/MIMICIV_sepsis.py
    def get_septic(self, sepsis_codes, base_outdir, mimic_dir):
        diag_df = PandasUtils.load_csv(FNAME_DIAGNOSES, mimic_dir)
        admissions_df = PandasUtils.load_csv(FNAME_ADMISSIONS, mimic_dir)
        patients_df = PandasUtils.load_csv(FNAME_PATIENTS, mimic_dir)
        hadm_ids = self.select_septic_hadm_id(diag_df, sepsis_codes)

        outdir = f"{base_outdir}/ethnicity/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.get_septic_notes(hadm_ids, admissions_df, patients_df, get_protected_label=get_ethnicity, outdir=outdir, mimic_dir=mimic_dir)

        outdir = f"{base_outdir}/sex/"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.get_septic_notes(hadm_ids, admissions_df, patients_df, get_protected_label=get_sex, outdir=outdir, mimic_dir=mimic_dir)

    def select_septic_hadm_id(self, diag_df, sepsis_codes):
        print("Getting septic HADM_IDs")
        hadm_ids = diag_df[diag_df['ICD9_CODE'].isin(sepsis_codes)]['HADM_ID']
        # print("Septic HADM_IDs \n", list(hadm_ids))
        return list(hadm_ids)

    def get_septic_notes(self, septic_hadm_ids, admissions_df, patients_df, get_protected_label, outdir, mimic_dir,
                         fname_notes=FNAME_NOTES):
        print("Loading notes csv")
        notes_df = PandasUtils.load_csv(fname_notes, mimic_dir)

        print("Removing error entries")
        prev_len = notes_df.shape[0]
        notes_df = notes_df[notes_df['ISERROR'] != 1]
        assert notes_df.shape[0] < prev_len, "None of the entries are removed"

        print("Removing leading and trailing spaces and converting text to lowercase")
        notes_df['TEXT'] = notes_df['TEXT'].str.strip()
        print("Converting text to lowercase")
        notes_df['TEXT'] = notes_df['TEXT'].str.lower()

        print("Removing blank and NA entries from TEXT and HADM_ID columns")
        notes_df['TEXT'].replace('', np.nan, inplace=True)
        notes_df.dropna(subset=['HADM_ID', 'TEXT'], inplace=True)

        print("Converting HADM ID to int")
        notes_df['HADM_ID'] = notes_df['HADM_ID'].astype('int64')
        print("Converting chartdate to datetime")
        notes_df['CHARTDATE'] = pd.to_datetime(notes_df['CHARTDATE'], format='%Y-%m-%d')

        print("Dropping duplicates")
        notes_df = notes_df.drop_duplicates(subset=['SUBJECT_ID', 'HADM_ID',
                                                    'CHARTDATE', 'CHARTTIME',
                                                    'CATEGORY', 'DESCRIPTION',
                                                    'TEXT'],
                                            keep='first')

        print("Adding septic labels")
        notes_df['SEPTIC'] = np.where(notes_df['HADM_ID'].isin(septic_hadm_ids),
                                      "septic", "non_septic")

        len_all_notes = [len(cur_note.split()) for cur_note in list(notes_df['TEXT'])]
        print("Average length of notes: ", statistics.mean(len_all_notes))
        print("Total number of notes: ", len(len_all_notes))

        print("Number of septic notes: ", notes_df[notes_df['SEPTIC'] == "septic"].shape[0])

        print("All categories of notes")
        print(set(notes_df['CATEGORY']))

        print("Removing social work notes")
        notes_df = notes_df[notes_df['CATEGORY'] != "Social Work"]

        print("Removing rehabilitation notes ")
        notes_df = notes_df[notes_df['CATEGORY'] != "Rehab Services"]

        print("Removing nutrition notes ")
        notes_df = notes_df[notes_df['CATEGORY'] != "Nutrition"]

        print("Removing discharge notes to prevent direct mention of rnn_expl_rules")
        notes_df = notes_df[notes_df['CATEGORY'] != "Discharge summary"]

        print("New categories, ", set(notes_df['CATEGORY']))

        print("Total Number of notes: ", notes_df.shape[0])
        print("Number of septic notes: ", notes_df[notes_df['SEPTIC'] == "septic"].shape[0])

        note_subset = notes_df.loc[notes_df.groupby('HADM_ID').CHARTDATE.idxmax()]
        print("Number of notes after selecting last note per admission: ", note_subset.shape[0])
        print("Number of septic notes after selecting last note per admission: ",
              note_subset[note_subset['SEPTIC'] == "septic"].shape[0])

        print("Adding demographic data")
        note_subset = note_subset.merge(admissions_df, how="left", on="HADM_ID")

        print("Adding patient data")
        note_subset = note_subset.merge(patients_df, left_on="SUBJECT_ID_x", right_on="SUBJECT_ID")

        print("Serializing data")
        dataset = []

        for hadm_id in note_subset['HADM_ID'].tolist():
            note_subset_hadm = note_subset[note_subset['HADM_ID'] == hadm_id]
            cur_label = note_subset_hadm['SEPTIC'].item()
            cur_protected_label = get_protected_label(note_subset_hadm)
            text = note_subset_hadm['TEXT'].item()
            dataset.append({"hadm_id": hadm_id, "x": text, "y": cur_label,
                            "protected_label": cur_protected_label})

        train_set, test_set = train_test_split(dataset, test_size=0.1, random_state=1)
        train_set, dev_set = train_test_split(train_set, test_size=0.1, random_state=1)
        save_json(train_set, f"{outdir}/train.json")
        save_json(dev_set, f"{outdir}/dev.json")
        save_json(test_set, f"{outdir}/test.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mimic_dir", dest='mimic_dir',
                        help="Path containing the MIMIC notes zipped files.",
                        required=True)
    parser.add_argument("--sepsis_out_dir", dest='sepsis_out_dir',
                        help="Path to write sepsis notes and labels to.",
                        required=True)

    args = parser.parse_args()

    mimic_dir = args.mimic_dir
    base_outdir = args.sepsis_out_dir

    sepsis_obj = SepsisMIMIC()
    sepsis_obj.get_septic([ICD9_SEPSIS, ICD9_SEVERE_SEPSIS, ICD9_SEPTIC_SHOCK], base_outdir, mimic_dir)
