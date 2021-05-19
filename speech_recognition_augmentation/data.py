import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random
from scipy.ndimage import rotate

# 음성 파일의 sample rate은 1초 = 16000으로 지정한다
SR = 16000


# 경진대회 전용 SpeechDataset 클래스를 정의한다
class SpeechDataset(Dataset):
    def __init__(self, mode, label_to_int, wav_list, label_list=None):
        self.mode = mode
        self.label_to_int = label_to_int
        self.wav_list = wav_list
        self.label_list = label_list
        self.sr = SR
        self.n_silence = int(len(wav_list) * 0.1)

        # 배경 소음 데이터를 미리 읽어온다
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob("/hdd1/hjinny/project/specaugment/kaggle/data/train/audio/_background_noise_/*.wav")]

    def get_one_word_wav(self, idx):
        # idx 번째 음성 파일을 1초만큼 읽어온다
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            wav = np.pad(wav, (0, self.sr - len(wav)), 'constant')
        return wav[:self.sr]

    def get_one_noise(self):
        # 배경 소음 데이터 중 랜덤하게 1초를 읽어온다
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        # num_noise 만큼의 배경 소음을 합성한다
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        # 배경 소음 데이터를 silence로 가정하고 불러온다
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)
    
    def timeshift(self, wav, ms=100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]
    
    def spec_augment(self, wav, T=40, F=15, time_mask_num=1, freq_mask_num=1):
        wav = preprocess_mel(wav)
        feat_size = wav.shape[0]
        seq_len = wav.shape[1]
        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=T)
            t = int(t)
            t0 = random.randint(0, seq_len - t)
            wav[t0 : t0 + t] = 0
        # freq mask
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=F)
            f = int(f)
            f0 = random.randint(0, feat_size - f)
            wav[:, f0 : f0 + f] = 0
        return wav
    
    def translate_specUp(img, shift=4, direction=1, roll=True, T=40, F=15, time_mask_num=2, freq_mask_num=2):
        assert direction in [1, 2], 'Directions should be left|right'
        direction = random.randint(1,2)
        img = img.copy()
        if direction == 1:
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:,:shift] = np.fliplr(right_slice)
        if direction == 2:
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        feat_size = img.shape[0]
        seq_len = img.shape[1]
        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=T)
            t = int(t)
            t0 = random.randint(0, seq_len - t)
            #wav[t0 : t0 + t] = 0
        # freq mask
            for _ in range(freq_mask_num):
                f = np.random.uniform(low=0.0, high=F)
                f = int(f)
                f0 = random.randint(0, feat_size - f)
                
                img[t0 : t0 + t, f0 : f0 + f] = 0        
        return img     
    
    def spec_augment_up(self, wav, T=40, F=15, time_mask_num=2, freq_mask_num=2):
        wav = preprocess_mel(wav)
        feat_size = wav.shape[0]
        seq_len = wav.shape[1]
        # time mask
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=T)
            t = int(t)
            t0 = random.randint(0, seq_len - t)
            #wav[t0 : t0 + t] = 0
        # freq mask
            for _ in range(freq_mask_num):
                f = np.random.uniform(low=0.0, high=F)
                f = int(f)
                f0 = random.randint(0, feat_size - f)
                
                wav[t0 : t0 + t, f0 : f0 + f] = 0
        return wav
    
    
    def translate(self, img, shift=3, direction='right', roll=True):
        img = preprocess_mel(img)
        assert direction in ['right', 'left'], 'Directions should be left|right'
        img = img.copy()
        if direction == 'right':
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:,:shift] = np.fliplr(right_slice)
        if direction == 'left':
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        return img


    
    def stretch(self, data, rate=1):
        input_length = 16000
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data
    
    def gaussian_noise(self,img, mean=0, sigma=3):
        img = img.copy()
        noise = np.random.normal(mean, sigma, img.shape)
        mask_overflow_upper = img+noise >= 1.0
        mask_overflow_lower = img+noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        img += noise
        return img
    
    def change_channel_ratio(self,wav, channel='r', ratio=0.5):
        img = preprocess_mel(wav)
        assert channel in 'rgb', "Value for channel: r|g|b"
        img = img.copy()
        ci = 'rgb'.index(channel)
        img[:, : ci] *= ratio
        return img

    def change_channel_ratio_gauss(self, wav, channel='r', mean=0, sigma=0.03):
        img = preprocess_mel(wav)
        assert channel in 'rgb', "cahenel must be r|g|b"
        img = img.copy()
        ci = 'rgb'.index(channel)
        img[:, : ci] = self.gaussian_noise(img[:, : ci], mean=mean, sigma=sigma)
        return img
    
# plot_grid([change_channel_ratio_gauss(img, mean=-0.01, sigma=0.1),
#            change_channel_ratio_gauss(img, mean=-0.05, sigma=0.1), 
#            change_channel_ratio_gauss(img, mean=-0.1, sigma=0.1)],
#            1, 3, figsize=(10, 5))

    def __len__(self):
        # 교차검증 모드일 경우에는 ‘silence’를 추가한 만큼이 데이터 크기이고, Test 모드일 경우에는 제공된 테스트 데이터가 전부이다
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    def __getitem__(self, idx):
        # idx번째 음성 데이터 하나를 반환한다
        if idx < len(self.wav_list):
            # 전처리는 mel spectrogram으로 지정한다
            # (옵션) 여기서 Data Augmentation을 수행할 수 있다.
            wav_numpy = preprocess_mel(self.get_one_word_wav(idx)) if self.mode != 'train' else self.get_specAug_up_imgwav(idx)
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            # 음성 스펙트로그램(spec), 파일 경로(id)와 정답값(label)을 반환한다
            if self.mode == 'test':
                return {'spec': wav_tensor, 'id': self.wav_list[idx]}
            else:
                label = self.label_to_int.get(self.label_list[idx], len(self.label_to_int))
                return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}
        else:
            # 배경 소음을 반환한다
            wav_numpy = preprocess_mel(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])))
            wav_tensor = torch.from_numpy(wav_numpy).float()
            wav_tensor = wav_tensor.unsqueeze(0)
            return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_to_int) + 1}
        
    def get_specAug_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.spec_augment(one_word_wav)
    
    def get_trans_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.translate(one_word_wav)   
    
    def get_translate_specUp_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        one_word_wav = preprocess_mel(one_word_wav)
        return self.translate_specUp(one_word_wav)       
    
    def get_specAug_up_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.spec_augment_up(one_word_wav)
    
    def get_stretch_wav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.stretch(one_word_wav)
    
    def get_gaussian_noise_wav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        one_word_wav = preprocess_mel(one_word_wav)
        return self.gaussian_noise(one_word_wav)    
    
    def get_change_channel_ratio_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.change_channel_ratio(one_word_wav)
    
    def get_change_channel_ratio_gauss_imgwav(self, idx):
        one_word_wav = self.get_one_word_wav(idx)
        return self.change_channel_ratio_gauss(one_word_wav,mean=-0.01, sigma=0.1)    
    
    
    def get_timeshift_wav(self, idx):
        # 음성 파형의 높이를 조정하는 scale
        scale = random.uniform(0.75, 1.25)
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        return scale * (self.timeshift(one_word_wav,shift_range))

    def get_noisy_wav(self, idx):
        # 음성 파형의 높이를 조정하는 scale 
        scale = random.uniform(0.75, 1.25)
        # 추가할 노이즈의 개수
        num_noise = random.choice([1, 2])
        # 노이즈 음성 파형의 높이를 조정하는 max_ratio
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        # 노이즈를 추가할 확률 mix_noise_proba
        mix_noise_proba = random.choice([0.1, 0.3])
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            # Data Augmentation을 수행한다.
            return scale * (self.get_mix_noises(num_noise,max_ratio))
        else:
            # 원본 음성 데이터를 그대로 반환한다.
            return one_word_wav 
        
    def get_noisyShift_wav(self, idx):
#         one_word_wav = self.get_one_word_wav(idx)
#         return self.spec_augment(one_word_wav)
        # 음성 파형의 높이를 조정하는 scale
        scale = random.uniform(0.75, 1.25)
        # 추가할 노이즈의 개수
        num_noise = random.choice([1, 2])
        # 노이즈 음성 파형의 높이를 조정하는 max_ratio
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        # 노이즈를 추가할 확률 mix_noise_proba
        mix_noise_proba = random.choice([0.1, 0.3])
        # 음성 데이터를 좌우로 평행이동할 크기 shift_range
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            # Data Augmentation을 수행한다.
            return scale * (self.timeshift(one_word_wav,shift_range)+self.get_mix_noises(num_noise,max_ratio))
        else:
            # 원본 음성 데이터를 그대로 반환한다.
            return one_word_wav
        
# mel spectrogram 전처리 함수이다
def preprocess_mel(data, n_mels=40):
    spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram

def preprocess_wav(data):
    wavdata = librosa.feature.inverse.mel_to_audio(data,sr=SR)
    wavdata = librosa.power_to_db(wavdata)
    wavdata = wavdata.astype(np.float32)
    return wavdata
