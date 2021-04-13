# my-tts-tools



## Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlaWod/my-tts-tools/blob/master/demo.ipynb)



## Usage

#### LJSpeech
```bash
python prepare_data.py config/LJSpeech/preprocess.yaml
```

```bash
./mfa/montreal-forced-aligner/bin/mfa_align ./preprocessed_data/LJSpeech/data ./mytext/lexicon/librispeech-lexicon.txt english ./preprocessed_data/LJSpeech/textgrid
```

```bash
python preprocess.py config/LJSpeech/preprocess.yaml
```

### AISHELL3
```bash
python prepare_data.py config/AISHELL3/preprocess.yaml
```

```bash
./mfa/montreal-forced-aligner/bin/mfa_train_and_align ./preprocessed_data/AISHELL3/data ./mytext/lexicon/pinyin-lexicon-r.txt ./preprocessed_data/AISHELL3/textgrid
```

```bash
python preprocess.py config/AISHELL3/preprocess.yaml
```