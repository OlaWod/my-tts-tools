import yaml
import argparse

from myaudio import TacotronSTFT
from myprocessor.process import process_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, 'r'), Loader = yaml.FullLoader)

    print("Loading STFT...")
    _stft = TacotronSTFT(
        config["stft"]["filter_length"],
        config["stft"]["hop_length"],
        config["stft"]["win_length"],
        config["mel"]["n_mel_channels"],
        config["audio"]["sampling_rate"],
        config["mel"]["mel_fmin"],
        config["mel"]["mel_fmax"],
    ) 

    process_data(config, _stft)

    '''
    !python preprocess.py config/LJSpeech/preprocess.yaml
    '''
