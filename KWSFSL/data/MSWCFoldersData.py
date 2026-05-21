# Modified from GSCSpeechData.py to support dynamic folder discovery

import os
from functools import partial
import glob
import hashlib
import math
import os.path
import random
import re
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose as compose
import torchaudio
import torch.nn.functional as F

from .data_utils import SetDataset, SimpleDataset

# settings
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0
RANDOM_SEED = 59185


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown=False, include_unknown=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        if fixed_silence_unknown:
            skip = 2
            fixed_class = torch.tensor([SILENCE_INDEX, UNKNOWN_WORD_INDEX])
            n_way = n_way-skip
            self.sampling = []
            for i in range(self.n_episodes):
                selected = torch.randperm(self.n_classes - skip)[:n_way]
                selected = torch.cat((fixed_class, selected.add(skip)))
                self.sampling.append(selected)
        else:
            self.sampling = [torch.randperm(self.n_classes)[
                :self.n_way] for i in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]


def prepare_words_list(wanted_words, silence, unknown):
    extra_words = []
    if silence:
        extra_words.append(SILENCE_LABEL)
    if unknown:
        extra_words.append(UNKNOWN_WORD_LABEL)
    return extra_words + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


class MSWCFoldersDataset:
    def __init__(self, data_dir, GSCtype, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration']
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(
            self.sample_rate * self.clip_duration_ms / 1000)

        self.use_background = args['include_noise']
        self.background_volume = args['bg_volume']
        self.background_frequency = args['bg_frequency']

        self.silence = args['include_silence']
        self.silence_num_samples = args['num_silence']
        self.unknown = args['include_unknown']

        self.data_cache = {}
        self.data_dir = data_dir

        params = {
            'silence_percentage': 10.0,
            'unknown_percentage': 10.0,
            'validation_percentage': 10.0,
            'testing_percentage': 10.0,
        }

        # Dynamically find all word folders or filter specific ones
        if GSCtype and GSCtype not in ['eval', 'train', 'val', 'all', '']:
            if ':' in GSCtype:
                all_folders = [w.strip() for w in GSCtype.split(':') if w.strip()]
            else:
                all_folders = [w.strip() for w in GSCtype.split(',') if w.strip()]
        else:
            all_folders = [f for f in os.listdir(data_dir) if os.path.isdir(
                os.path.join(data_dir, f)) and not f.startswith('_')]
        print(f"Found {len(all_folders)} word folders in {data_dir}")

        params['wanted_words'] = all_folders
        # No explicit unknown words list for MSWC folders
        params['unknown_words'] = []

        self.generate_data_dictionary(params)
        self.background_data = self.load_background_data()
        self.cuda = cuda
        self.max_class = len(all_folders)

    def get_episodic_fixed_sampler(self, num_classes,  n_way, n_episodes, fixed_silence_unknown=False, include_unknown=True):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes, fixed_silence_unknown=fixed_silence_unknown, include_unknown=include_unknown)

    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic',
                                include_silence=True, include_unknown=True, unique_speaker=False):

        class_list = []
        for item in self.words_list:
            if not include_silence and item == SILENCE_LABEL:
                continue
            if not include_unknown and item == UNKNOWN_WORD_LABEL:
                continue
            class_list.append(item)

        if sampler == 'episodic':
            sampler = self.get_episodic_fixed_sampler(len(class_list),
                                                      n_way, n_episodes)

        dl_list = []
        if set_index in ['training', 'testing']:
            for keyword in class_list:
                # Direct access to pre-filtered class list
                samples_for_class = self.data_set[set_index].get(keyword, [])
                ts_ds = self.get_transform_dataset_from_list(samples_for_class)
                
                dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples,
                                                 shuffle=True, num_workers=0)
                dl_list.append(dl)

            ds = SetDataset(dl_list)
            data_loader_params = dict(batch_sampler=sampler,  num_workers=8,
                                      pin_memory=not self.cuda)
            dl = torch.utils.data.DataLoader(ds, **data_loader_params)
        else:
            raise ValueError(
                "Set index = {} in episodic dataset is not correct.".format(set_index))

        return dl

    def get_iid_dataloader(self, set_index, batch_size, class_list=False, include_silence=True, include_unknown=True, unique_speaker=False):
        if not class_list:
            class_list = []
            for item in self.words_list:
                if not include_silence and item == SILENCE_LABEL:
                    continue
                if not include_unknown and item == UNKNOWN_WORD_LABEL:
                    continue
                class_list.append(item)

        # Flatten the dictionary for IID loading
        flattened_list = []
        for keyword in class_list:
            flattened_list.extend(self.data_set[set_index].get(keyword, []))

        ts_ds = self.get_transform_dataset_from_list(flattened_list)
        dl = torch.utils.data.DataLoader(
            ts_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        return dl

    def get_transform_dataset_from_list(self, file_list):
        transforms = compose([
            partial(self.load_audio, 'file', 'label', 'data'),
            partial(self.adjust_volume, 'data'),
            partial(self.shift_and_pad, 'data'),
            partial(self.mix_background, self.use_background, 'data', 'label'),
            partial(self.label_to_idx, 'label', 'label_idx')
        ])
        ls_ds = ListDataset(file_list)
        ts_ds = TransformDataset(ls_ds, transforms)
        return ts_ds

    def num_classes(self):
        return len(self.words_list)

    def label_to_idx(self, k, key_out, d):
        label_index = self.word_to_index[d[k]]
        d[key_out] = torch.LongTensor([label_index]).squeeze()
        return d

    def mix_background(self, use_background, k, key_label, d):
        foreground = d[k]
        if use_background or d[key_label] == SILENCE_LABEL:
            if not self.background_data:
                background_reshaped = torch.zeros(1, self.desired_samples)
                bg_vol = 0
            else:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(
                    0, len(background_samples) - self.desired_samples)
                background_clipped = background_samples[background_offset:(
                    background_offset + self.desired_samples)]
                background_reshaped = background_clipped.reshape(
                    [1, self.desired_samples])
                if np.random.uniform(0, 1) < self.background_frequency:
                    bg_vol = np.random.uniform(0, self.background_volume)
                else:
                    bg_vol = 0
        else:
            background_reshaped = torch.zeros(1, self.desired_samples)
            bg_vol = 0

        background_mul = background_reshaped * bg_vol
        background_add = background_mul + foreground
        background_clamped = torch.clamp(background_add, -1.0, 1.0)
        d[k] = background_clamped
        return d

    def load_background_data(self):
        background_path = os.path.join(
            self.data_dir, '_background_noise_', '*.wav')
        background_data = []
        for wav_path in glob.glob(background_path):
            bg_sound, bg_sr = torchaudio.load(wav_path)
            background_data.append(bg_sound.flatten())
        return background_data

    def shift_and_pad(self, key, d):
        audio = d[key]
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        time_shift_amount = np.random.randint(-time_shift,
                                              time_shift) if time_shift > 0 else 0

        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount

        audio_len = audio.size(1)
        if audio_len < self.desired_samples:
            pad = (0, self.desired_samples-audio_len)
            audio = F.pad(audio, pad, 'constant', 0)

        padded_foreground = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced_foreground = torch.narrow(
            padded_foreground, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced_foreground
        return d

    def adjust_volume(self, key, d):
        d[key] = d[key] * self.foreground_volume
        return d

    def load_audio(self, key_path, key_label, out_field, d):
        sound, _ = torchaudio.load(
            uri=d[key_path], normalize=True, num_frames=self.desired_samples)
        if d[key_label] == SILENCE_LABEL:
            sound.zero_()
        d[out_field] = sound
        return d

    def generate_data_dictionary(self, training_parameters):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        wanted_words_index = {}

        skip = 0
        if self.silence:
            skip += 1
        if self.unknown:
            skip += 1

        for index, wanted_word in enumerate(training_parameters['wanted_words']):
            wanted_words_index[wanted_word] = index + skip

        # Store as dictionary of lists: {set: {word: [samples]}}
        self.data_set = {'validation': {}, 'testing': {}, 'training': {}}
        all_words = {}
        search_path = os.path.join(self.data_dir, '*', '*.wav')

        for wav_path in glob.glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            if word == BACKGROUND_NOISE_LABEL:
                continue

            speaker_id = os.path.basename(wav_path).split('_')[0]
            all_words[word] = True
            set_index = which_set(
                wav_path, training_parameters['validation_percentage'], training_parameters['testing_percentage'])

            if word in wanted_words_index:
                if word not in self.data_set[set_index]:
                    self.data_set[set_index][word] = []
                self.data_set[set_index][word].append(
                    {'label': word, 'file': wav_path, 'speaker': speaker_id})

        # Handle silence samples
        if self.silence:
            # Find a fallback wav for silence generation
            silence_wav_path = None
            for s_idx in ['training', 'validation', 'testing']:
                for w in self.data_set[s_idx]:
                    if self.data_set[s_idx][w]:
                        silence_wav_path = self.data_set[s_idx][w][0]['file']
                        break
                if silence_wav_path: break

            if silence_wav_path:
                for set_index in ['validation', 'testing', 'training']:
                    # Calculate total samples in this set to determine silence count
                    set_size = sum(len(v) for v in self.data_set[set_index].values())
                    silence_size = int(math.ceil(set_size * training_parameters['silence_percentage'] / 100))
                    
                    self.data_set[set_index][SILENCE_LABEL] = []
                    for _ in range(silence_size):
                        self.data_set[set_index][SILENCE_LABEL].append(
                            {'label': SILENCE_LABEL, 'file': silence_wav_path, 'speaker': "None"})

        # Shuffle each class list
        for set_index in ['validation', 'testing', 'training']:
            for word in self.data_set[set_index]:
                rand_data_order = random.Random(RANDOM_SEED)
                rand_data_order.shuffle(self.data_set[set_index][word])

        self.words_list = prepare_words_list(
            training_parameters['wanted_words'], self.silence, self.unknown)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
        if self.silence:
            self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
        if self.unknown:
            self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX
