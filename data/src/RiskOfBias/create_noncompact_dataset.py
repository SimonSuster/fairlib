from fairlib.src.dataloaders.loaders.RoB import label_map, get_protected_group, get_protected_label_map
from fairlib.src.utils import load_json, save_json


def load_data(data_dir, split):
    protected_labels = []
    xs = []
    ys = []
    pmids = []

    data_path = f"{data_dir}/{split}.json"
    protected_group = get_protected_group(data_dir)
    protected_label_map = get_protected_label_map(protected_group)

    for c, i in enumerate(load_json(data_path)):
        i_protected_labels = i["protected_labels"]
        if not isinstance(i_protected_labels, list):
            i_protected_labels = [i_protected_labels]

        # when multiple topics exist, create one instance per each topic:
        for protected_label in i_protected_labels:
            p_label = protected_label.replace("_", " ")
            if protected_label_map.get(p_label, None) is None:
                continue

            protected_labels.append(p_label)
            abstract = i["abstract"]
            xs.append(abstract)
            ys.append(int(label_map(i["labels"])))
            pmids.append(i["pmid"])

    return xs, ys, protected_labels, pmids


def main(data_dir):
    for split in ["train", "dev", "test"]:
        dataset = []
        xs, ys, protected_labels, pmids = load_data(data_dir, split)
        for c, (x, y, protected_label, pmid) in enumerate(zip(xs, ys, protected_labels, pmids)):
            dataset.append(
                {"id": c, "pmid": pmid, "x": x, "y": y, "protected_label": protected_label}
            )
        out_f = f"{data_dir}/{split}_noncompact.json"
        save_json(dataset, out_f)
        print(f"{split} set written to {out_f}")


if __name__ == "__main__":
    main(data_dir="/home/simon/Apps/robotreviewer/rob_abstract_dataset_area/")
    main(data_dir="/home/simon/Apps/robotreviewer/rob_abstract_dataset_sex/")
