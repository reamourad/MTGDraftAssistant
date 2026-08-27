import os

import pandas
import json
from pathlib import Path
import random

from data_processing.config import DATA_DIR, CARD_LIST_FILENAME

SET_NAME = "MH3"
FOLDER = f"{DATA_DIR}/{SET_NAME}"

def unpack_csv_to_card_list(csv_path, file_destination):
    #0. find the .csv.gz file name
    csv_file_matches = list(Path(csv_path).glob("*.csv.gz"))
    if not csv_file_matches:
        return None

    #1. we want to read the .csv.gz
    column_names = pandas.read_csv(csv_file_matches[0], nrows=0).columns

    #2. we want to get all the possible cards that can get drafted and save this as a list in /data
    card_names = set()
    for col in column_names:
        if col.startswith('pack_card_'):
            card_name = col.replace('pack_card_', '')
            card_names.add(card_name)

    #Save this to file_destination
    file_name = os.path.join(file_destination, CARD_LIST_FILENAME)

    with open(file_name, 'w', newline='') as jsonfile:
        json.dump(sorted(card_names), jsonfile)

    return card_names

def pick_one(card_list):
    #for now all cards have the same probability

    # Load the JSON data
    with open(card_list, "r") as file:
        items = json.load(file)

    # Pick one random item from the list
    random_item = random.choice(items)
    return random_item

def pick_multiple(card_list, number):
    pack = []
    for num in range(number):
       pack.append(pick_one(card_list))
    return pack


if __name__ == "__main__":
    folder = FOLDER + "/" + CARD_LIST_FILENAME
    unpack_csv_to_card_list(FOLDER, FOLDER)
    result = pick_one(folder)
    print(FOLDER)
    print(result)