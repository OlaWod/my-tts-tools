import os
from tqdm import tqdm

import myaudio
import mytext


def process_data(config, _stft):
    data_dir = os.path.join(config['path']['feature_dir'], 'data')
    tg_dir = os.path.join(config['path']['feature_dir'], 'textgrid')
    
    mel_dir = os.path.join(config['path']['feature_dir'], 'mel')
    energy_dir = os.path.join(config['path']['feature_dir'], 'energy')
    f0_dir = os.path.join(config['path']['feature_dir'], 'f0')
    duration_dir = os.path.join(config['path']['feature_dir'], 'duration')

    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)
    os.makedirs(f0_dir, exist_ok=True)
    os.makedirs(duration_dir, exist_ok=True)

    max_wav_value = config['audio']['max_wav_value']
    hop_length = config['stft']['hop_length']

    infos = []
    
    print('Start processing...')

    for speaker in os.listdir(data_dir):
        print(speaker)
        for wav_name in os.listdir(os.path.join(data_dir, speaker)):
            if ".wav" not in wav_name:
                continue
            basename = wav_name.split('.')[0]

            wav_path = os.path.join(data_dir, speaker, wav_name)
            text_path = os.path.join(data_dir, speaker, "{}.lab".format(basename))
            tg_path = os.path.join(tg_dir, speaker, "{}.TextGrid".format(basename))
            if not os.path.exists(text_path) or not os.path.exists(tg_path):
                continue

            # get features
            with open(text_path, "r") as f:
                text = f.readline().strip("\n")
            phone, duration = mytext.process_textgrid(tg_path)
            idphone = mytext.phone_to_sequence(phone, config['text_cleaner'])
        
            info = "|".join([basename, speaker, text, phone, str(idphone)])
            infos.append(info)
        
            mel, energy = myaudio.get_mel_energy_from_wav(wav_path, _stft, max_wav_value)
            f0 = myaudio.get_f0_from_wav(wav_path, hop_length, max_wav_value)

            # save files
            myaudio.save_feature_to_npy(duration, 'duration', duration_dir, basename)
            myaudio.save_feature_to_npy(mel, 'mel', mel_dir, basename)
            myaudio.save_feature_to_npy(energy, 'energy', energy_dir, basename)
            myaudio.save_feature_to_npy(f0, 'f0', f0_dir, basename)

    # Write metadata
    with open(os.path.join(config['path']['feature_dir'], "train.txt"), "w", encoding="utf-8") as f:
        for info in infos[config['val_size'] :]:
            f.write(info + "\n")
    with open(os.path.join(config['path']['feature_dir'], "val.txt"), "w", encoding="utf-8") as f:
        for info in infos[: config['val_size']]:
            f.write(info + "\n")
