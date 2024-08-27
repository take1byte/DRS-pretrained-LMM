import pandas as pd

from pathlib import Path
import os
from example_generator import ExampleGenerator

DATA = Path(os.path.expandvars("$HOME/data/drs"))
SENTENCES_CSV = "test_sentences.csv"
SENTENCES_TXT = "test_sentences_enXX.txt"
DRS_OUT = "test_sentences_drs.txt"
SENTENCES_ROUNDTRIP = "test_sentences_roundtrip.txt"


def prepare_sentences(
    sentences_in: Path = DATA / SENTENCES_CSV,
    sentences_out: Path = DATA / SENTENCES_TXT,
) -> None:
    sents = pd.read_csv(sentences_in, header=0)
    sents[["Sentence"]].to_csv(sentences_out, header=None, index=False)  # type: ignore


def parse_sentences(sentences_txt: Path = DATA / SENTENCES_TXT) -> None:
    eg = ExampleGenerator()
    sents = []

    with open(sentences_txt, "r") as f:
        sents = f.readlines()

    with open(DATA / DRS_OUT, "w") as drs_file:
        with open(DATA / SENTENCES_ROUNDTRIP, "w") as roundtrip_file:
            for sent in sents:
                sent = sent.strip('"\n')
                drs = eg.en2drs(sent)
                roundtrip = eg.drs2en(drs)
                drs_file.write(drs + "\n")
                roundtrip_file.write(roundtrip + "\n")
                print(f"TXT: {sent}\nDRS: {drs}\nROUNDTRIP TXT: {roundtrip}\n")


def main():
    parse_sentences()


if __name__ == "__main__":
    main()
