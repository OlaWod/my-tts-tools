import re
import tgt
import numpy as np
from string import punctuation
from g2p_en import G2p
from pypinyin import pinyin, Style

from .utils import phone_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def process_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon('mytext/lexicon/librispeech-lexicon.txt')

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
   
    sequence = np.array(
        phone_to_sequence(
            phones, ["english_cleaners"]
        )
    )
    
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    print(sequence)

    return phones, np.array(sequence)


def process_mandarin(text):
    lexicon = read_lexicon('mytext/lexicon/pinyin-lexicon-r.txt')

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    
    sequence = np.array(
        phone_to_sequence(
            phones, []
        )
    )

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    print(sequence)

    return phones, np.array(sequence)


def process_textgrid(tg_path):
    textgrid = tgt.io.read_textgrid(tg_path)
    tier = textgrid.get_tier_by_name("phones")

    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(e-s)

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return "{" + " ".join(phones) + "}", durations
