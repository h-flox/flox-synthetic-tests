import re
import pandas as pd

from collections import defaultdict
from typing import Any


def process_line(line: str) -> dict[str, Any]:
    record = {}

    # Grab and process the data in the brackets from the original text output.
    for piece in re.findall(r"\[.*?\]", line):
        piece = piece.translate(str.maketrans({"[": "", "]": ""}))
        name, data = piece.split("=")
        record[name] = int(data)

    # Grab the throughput which is formatted differently.
    last_piece = line.split("]")[-1]
    try:
        pieces = last_piece.split()
    except IndexError as err:
        print(last_piece)
        raise err

    if pieces[0] == "Launched":
        record["event_kind"] = "launched"
        record["timing_sec"] = float(pieces[-1].replace("s", ""))
    elif pieces[0] == "Finished":
        record["event_kind"] = "finished"
        record["timing_sec"] = float(pieces[-1].replace("s", ""))
    elif pieces[0] == "Throughput":
        record["event_kind"] = "throughput"
        record["timing_sec"] = float(pieces[1])
    else:
        raise ValueError

    return record


def process(txt: str) -> pd.DataFrame:
    records = []
    for line in txt.split("\n"):
        records.append(process_line(line))

    df = pd.DataFrame.from_records(records)
    return df


def load_txt(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


if __name__ == "__main__":
    txt = load_txt("all.txt")
    df = process(txt)
    print(df.head())
