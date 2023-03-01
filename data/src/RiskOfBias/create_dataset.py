import argparse
import os

from sklearn.model_selection import train_test_split

from fairlib.src.utils import load_json, save_json, get_file_list


def get_abstract(pmid, input_dir):
    with open(f"{input_dir}/abstracts/{pmid}.txt") as fh:
        return " ".join([l.strip() for l in fh])


def main(input_dir, label_f, protected_label_f, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pmids_abstracts = {os.path.splitext(os.path.basename(fn))[0] for fn in get_file_list(f"{input_dir}/abstracts/")}
    labels = load_json(label_f)
    protected_labels = load_json(protected_label_f)
    protected_labels = {k: v for k, v in protected_labels.items() if v is not None}

    # assert len(labels.keys() - protected_labels.keys()) == 0
    pmids = list(labels.keys() & pmids_abstracts & protected_labels.keys())
    pmids_train, pmids_test = train_test_split(pmids, test_size=0.1, random_state=1)
    pmids_train, pmids_dev = train_test_split(pmids_train, test_size=0.1, random_state=1)
    pmids_split = {"train": pmids_train, "dev": pmids_dev, "test": pmids_test}

    # split pmids into train/dev/test

    dataset = {
        "train": [],
        "dev": [],
        "test": []
    }

    # write data
    for split, pmids in pmids_split.items():
        for pmid in pmids:
            abstract = get_abstract(pmid, input_dir)
            if len(set(labels[pmid][0].values()) - {"Unclear risk", "Low risk", "High risk"}) != 0:
                print(f"Skipping {pmid} due to unknown labels.")
                continue
            dataset[split].append({"pmid": pmid, "abstract": abstract, "labels": labels[pmid][0],
                                   "protected_labels": protected_labels[pmid]})

    save_json(dataset["train"], f"{output_dir}/train.json")
    print(f"Train set written to {output_dir}/train.json")
    save_json(dataset["dev"], f"{output_dir}/dev.json")
    print(f"Dev set written to {output_dir}/dev.json")
    save_json(dataset["test"], f"{output_dir}/test.json")
    print(f"Test set written to {output_dir}/test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', default="/home/simon/Apps/robotreviewer/pubmed_abstracts/",
                        help='input dir: title and abtract files')
    parser.add_argument('-label_f', default="/home/simon/Apps/SysRevData/data/dataset/robotreviewer_train_data.json",
                        help='json file containing labels per each pmid')
    parser.add_argument('-protected_label_f',
                        default="/home/simon/Apps/SysRevData/data/dataset/robotreviewer_topics_train.json",
                        help='json file containing protected labels per each pmid')
    parser.add_argument('-output_dir',
                        help="directory containing prepared data")
    args = parser.parse_args()

    main(args.input_dir, args.label_f, args.protected_label_f, args.output_dir)
