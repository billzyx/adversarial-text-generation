import datasets
import re

import dataloaders_adress


def keep_letters_only(input_string):
    return re.sub(r'[^a-zA-Z]', '', input_string).lower()


def difference_by_key(list1, list2, key):
    set1 = {keep_letters_only(d[key]) for d in list1 if key in d}
    set2 = {keep_letters_only(d[key]) for d in list2 if key in d}

    # Dictionaries that are in list1 but not in list2
    diff1 = [d for d in list1 if keep_letters_only(d.get(key)) not in set2]

    # Dictionaries that are in list2 but not in list1
    diff2 = [d for d in list2 if keep_letters_only(d.get(key)) not in set1]

    return diff1, diff2


def load_dataset_easy(dataset_str):
    if dataset_str.startswith('ADReSS'):
        return dataloaders_adress.load_dataset_easy(dataset_str)
    else:
        dataset_split = dataset_str.split('-')
        return datasets.load_dataset(''.join(dataset_split[:-1]), split=dataset_split[-1])
