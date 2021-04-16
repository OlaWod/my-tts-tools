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

**prepare_data:**
```bash
python prepare_data.py config/AISHELL3/preprocess.yaml
```

**mfa:**
```bash
./mfa/montreal-forced-aligner/bin/mfa_train_and_align ./preprocessed_data/AISHELL3/data ./mytext/lexicon/pinyin-lexicon-r.txt ./preprocessed_data/AISHELL3/textgrid
```

**preprocess:**
```bash
python preprocess.py config/AISHELL3/preprocess.yaml
```

**train:**
```bash
python train.py -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

**evaluate:**
```bash
python evaluate.py -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

**synthesize:**
```bash
python synthesize.py --source test.txt --restore_step 3000 -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```