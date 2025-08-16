import re
from pathlib import Path
import config

def preprocess_data():
    # SMHD Processing
    smhd_labels_path = [config.SMHD_SAVE_PATH / "SMHD.txt", config.SMHD_SAVE_PATH / "Class_Notes_SMHD.txt"]

    png_ids = {p.stem for p in config.SMHD_SAVE_PATH.glob("*/*.png")}

    punct_fix = re.compile(r'\s+([,.;:!?])') # strip spaces before punctuation
    smhd_img_ids, smhd_labels = [], []
    seen = set()

    for path in smhd_labels_path:
        with path.open() as f:
            for raw in f:
                if ',' not in raw: # malformed line → skip
                    continue

                img_id, rest = raw.rstrip('\n').split(',', 1)

                if img_id not in png_ids:
                    continue

                if img_id in seen:
                    continue
                seen.add(img_id)

                parts = rest.lstrip().split(' ', 1)
                label = parts[1] if len(parts) == 2 else rest.lstrip()
                label = punct_fix.sub(r'\1', label)

                smhd_img_ids.append(img_id)
                smhd_labels.append(label)

    # print(len(smhd_img_ids))

    # print(smhd_labels[1960:1970])
    # print(smhd_img_ids[1960:1970])

    # IAM Processing
    iam_truth_path = "./datasets/lines/lines.txt"

    with open(iam_truth_path, "r") as f:
        lines = f.readlines()
        #iam_paths = [line.split()[0] for line in lines if not line.startswith('#')]
        iam_labels = [line.split()[-1].replace("|"," ") for line in lines if not line.startswith('#')]

    return smhd_labels, iam_labels

def preprocess_testing_data():
    test_truth_path = "./Datasets/test_lines/Labels.txt"

    with open(test_truth_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        test_labels = [line.strip().split('\t')[1] for line in lines]

    return test_labels

if __name__ == "__main__":
    print(preprocess_testing_data())