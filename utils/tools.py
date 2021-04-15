import torch
import numpy as np


def to_device(data, device):
    (
        basenames,
        speaker_ids,
        phone_ids,

        mels,
        energys,
        f0s,
        durations,

        phoneid_lens,
        max_phoneid_lens,
        mel_lens,
        max_mel_lens
    ) = data

    speaker_ids = torch.from_numpy(speaker_ids).long().to(device)
    phone_ids = torch.from_numpy(phone_ids).long().to(device)
    mels = torch.from_numpy(mels).float().to(device)
    energys = torch.from_numpy(energys).float().to(device)
    f0s = torch.from_numpy(f0s).float().to(device)
    durations = torch.from_numpy(durations).long().to(device)
    phoneid_lens = torch.from_numpy(phoneid_lens).to(device)
    mel_lens = torch.from_numpy(mel_lens).to(device)

    return (
        basenames,
        speaker_ids,
        phone_ids,

        mels,
        energys,
        f0s,
        durations,

        phoneid_lens,
        max_phoneid_lens,
        mel_lens,
        max_mel_lens
    )


def pad_1D(inputs):
    def pad(x, max_len, PAD=0):     
        x_padded = np.pad(
            x, (0, max_len - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad(x, max_len) for x in inputs])

    return padded


def pad_2D(inputs):
    def pad(x, max_len, PAD=0):
        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:,:s]

    max_len = max(np.shape(x)[0] for x in inputs)
    padded = np.stack([pad(x, max_len) for x in inputs])

    return padded
