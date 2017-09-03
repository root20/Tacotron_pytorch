# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from multiprocessing import Process, Queue
import os, re, librosa, argparse, torch, pickle, multiprocessing
import numpy as np

def get_vocab(dataset, directory):
    """ read files & create characters dictionary
        Decide what characters to remove at this stage
    """
    if dataset == 'blizzard':
        preprocess_text_blizzard(directory, 'prompts.gui')
    elif dataset == 'nancy':
        preprocess_text_nancy(directory, 'prompts.data')
    elif dataset == 'etri':
        preprocess_text_etri(directory, 'prompts.data')

    vocabs = set([])

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename[-4:] == '.txt':
                path = os.path.join(root, filename)
                with open(path, 'r') as rFile:
                    line = rFile.readline()
                    while line:
                        line = line.strip()
                        for i in range(len(line)):
                            vocabs.add(line[i])
                        line = rFile.readline()
    vocabs = sorted(list(vocabs))
    print(vocabs)


def cleanse(directory, dir_bin, write_csv, spec_file_list):
    """ read files & cleanse each line
        look at cleansed file and check what to cleanse more
    """
    text_file_list = []
    cleansed_files = []
    vocabs = set([])
    vocabs.add('Z')         # there is no 'Z' in blizzard corpus (may cause problem later)

    dirts = ['`', '#', '@', '\|']
    dirts = '(' + '|'.join(dirts) + ')'
    reg_dirts = re.compile(dirts)
    reg_spaces = re.compile(r'\s+')
    reg_spacedSymbols = re.compile(r' (?P<ssym>\W)')

    for root, dirnames, filenames in os.walk(directory):
        wFileName = root[len(directory):]
        writePath = dir_bin + wFileName + '.txt'
        with open(writePath, 'w') as wFile:
            for filename in sorted(filenames):
                if filename[-4:] == '.txt':
                    readPath = os.path.join(root, filename)

                    with open(readPath, 'r') as rFile:
                        line = rFile.readline()
                        while line and len(line) > 0:
                            line = line.strip()
                            line = reg_dirts.sub('', line)
                            line = reg_spaces.sub(' ', line)
                            line = reg_spacedSymbols.sub(r'\g<ssym>', line)

                            wFile.write(line+'\n')

                            if write_csv:
                                for i in range(len(line)):
                                    vocabs.add(line[i])

                            line = rFile.readline()
        if write_csv:
            cleansed_files.append((writePath, wFileName))

    if write_csv:
        print('Start to write text binary files')
        if spec_file_list is None:
            print('spec_file_list is not found.')

        tmp_vocab = sorted(list(vocabs))
        vocab_dict = {}
        for i, vocab in enumerate(tmp_vocab):
            vocab_dict[vocab] = i + 1                           # zero will be used for null-padding
        torch.save(vocab_dict, dir_bin + 'vocab.t7')

        write_path = dir_bin + 'text.bin'
        offset = 0
        with open(write_path, 'wb') as w_file:
            for cleansed_tuple in cleansed_files:
                cleansed_file = cleansed_tuple[0]

                if spec_file_list is not None:
                    with open(cleansed_file, 'r') as rFile:
                        count = 0
                        line = rFile.readline().strip()
                        while line:
                            # print(line)
                            line = [vocab_dict[y] for y in line]

                            binary_text = pickle.dumps(line, protocol=pickle.HIGHEST_PROTOCOL)
                            w_file.write(binary_text)
                            file_id = spec_file_list[count][0]
                            text_file_list.append((file_id, len(line), offset, len(binary_text)))
                            offset += len(binary_text)

                            line = rFile.readline().strip()
                            count += 1
                else:
                    wFileName = cleansed_tuple[1]
                    writePath = dir_bin + wFileName + '_text.csv'

                    with open(writePath, 'w') as wFile:
                        with open(cleansed_file, 'r') as rFile:
                            line = rFile.readline().strip()
                            while line:
                                # print(line)
                                line = [str(vocab_dict[y]) for y in line]
                                wFile.write(','.join(line)+'\n')
                                line = rFile.readline().strip()

    return text_file_list


def preprocess_text_blizzard(directory, file):
    readPath = directory + file
    writePath = directory + '/prompts.txt'

    with open(writePath, 'w') as wFile:
        with open(readPath, 'r') as rFile:
            line = rFile.readline()     # 1st line
            while line and len(line) > 0:
                line = rFile.readline()     # 2nd line (txt included)
                wFile.write(line)

                rFile.readline()            # 3rd line
                line = rFile.readline()     # 1rd line of next wav file


def preprocess_text_nancy(directory, file):
    readPath = directory + file
    writePath = directory + '/prompts.txt'

    with open(writePath, 'w') as wFile:
        with open(readPath, 'r') as rFile:
            line = rFile.readline()
            while line and len(line) > 0:
                wFile.write(line[line.find('"')+1:-4].strip()+'\n')
                line = rFile.readline()


def preprocess_text_etri(directory, file):
    """
    Code from: https://github.com/neotune/python-korean-handler
    """
    readPath = directory + '/prompts.data'
    writePath = directory + '/prompts.txt'

    # 유니코드 한글 시작 : 44032, 끝 : 55199
    Start_Code, ChoSung, JungSung = 44032, 588, 28

    # 초성 리스트. 00 ~ 18
    ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    # 중성 리스트. 00 ~ 20
    JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']

    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    with open(writePath, 'w') as wFile:
        with open(readPath, 'r', encoding="utf-8") as rFile:
            line = rFile.readline()     # skip this line (utf8 header)
            line = rFile.readline()
            while line:
                line_dec = ""
                line = line.strip().split('\t')[1]
                line = list(line)

                for keyword in line:
                    # 한글 여부 check 후 분리: ㄱ~ㅎ + ㅏ~ㅣ+ 가~힣
                    if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
                        char_code = ord(keyword) - Start_Code
                        char1 = int(char_code / ChoSung)
                        line_dec += ChoSung_LIST[char1]
                        #print('초성 : {}'.format(CHOSUNG_LIST[char1]))
                        char2 = int((char_code - (ChoSung * char1)) / JungSung)
                        line_dec += JungSung_LIST[char2]
                        #print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
                        char3 = int((char_code - (ChoSung * char1) - (JungSung * char2)))
                        line_dec += JongSung_LIST[char3]
                        #print('종성 : {}'.format(JONGSUNG_LIST[char3]))
                    else:
                        line_dec += keyword

                wFile.write(line_dec + '\n')
                line = rFile.readline()


def trim_silence(audio, sr, frame_shift_inMS, file_name=None, beginning_buffer=2, ending_buffer=5):
    # # buffers are counted in number of frame shifts
    # onset = librosa.onset.onset_detect(audio, sr=sr)
    # if not onset:
    #     print('maybe empty file:', file_name)
    # onset_sample = librosa.frames_to_samples(onset)
    #
    # unit = sr * frame_shift_inMS / 1000
    # start_idx = onset_sample[0] - beginning_buffer * unit
    # if len(onset_sample) == 1:
    #     end_idx = -1
    # else:
    #     end_idx = onset_sample[-1] + ending_buffer * unit
    #
    # return audio[start_idx:end_idx]
    return audio


def preprocess_spec(dataset, f_type, rDirectory, dir_bin, q):
    if dataset == 'vctk':
        silence_threshold = 0.005
        sample_rate = 24000
    elif dataset == 'blizzard':
        silence_threshold = 0.005
        sample_rate = 16000
    elif dataset == '10':
        silence_threshold = 0.005
        sample_rate = 16000
    elif dataset == 'nancy':
        silence_threshold = 0.005
        sample_rate = 16000
    elif dataset == 'etri':
        silence_threshold = 0.005
        sample_rate = 16000

    frame_len_inMS = 50
    frame_shift_inMS = 12.5
    isMono      = True
    type_filter = f_type

    # params for stft
    n_fft = 2048
    window_len = int(np.ceil(frame_len_inMS * sample_rate / 1000))
    hop_length = int(np.ceil(frame_shift_inMS * sample_rate / 1000))

    # params for mel-filter
    mel_dim = 80
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=mel_dim)

    # params for normalization
    ref_level_db = 20
    min_level_db = -100


    files = []
    count = 0
    print('Check files..')
    for root, dirnames, filenames in os.walk(rDirectory):
        for filename in sorted(filenames):
            if filename[-4:] == '.wav':
                path = os.path.join(root, filename)
                audio,_ = librosa.load(path, sr=sample_rate, mono=isMono)
                audio = trim_silence(audio, sample_rate, frame_shift_inMS, filename=filename)
                length = len(audio)
                files.append((path, str(count), int(length)))
                count += 1
                # librosa.output.write_wav(path, audio, 16000)

    spec_max, spec_min = None, None

    if type_filter == 'linear':
        write_path = dir_bin + 'specL.bin'
    elif type_filter == 'mel':
        write_path = dir_bin + 'specM.bin'
    else:
        write_path = None

    print('Start writing %s spectrogram binary files' % f_type)
    spec_list = []
    offset = 0
    with open(write_path, 'wb') as w_file:
        for item in files:
            path = item[0]
            line_idx = item[1]

            audio,_ = librosa.load(path, sr=sample_rate, mono=isMono)
            audio = trim_silence(audio, sample_rate, frame_shift_inMS)

            D = librosa.stft(audio, n_fft=n_fft, win_length=window_len, window='hann', hop_length=hop_length)
            spec = np.abs(D)

            if type_filter == 'mel':
                # mel-scale spectrogram generation
                spec = np.dot(mel_basis, spec)
                spec = 20 * np.log10(np.maximum(1e-5, spec))
            elif type_filter == 'linear':
                # linear spectrogram generation
                spec = 20 * np.log10(np.maximum(1e-5, spec)) - ref_level_db

            # normalize
            spec = np.clip(-(spec - min_level_db) / min_level_db, 0, 1)
            spec = spec.T   # H x T -> T x H

            # write to file
            binary_spec = pickle.dumps(spec, protocol=pickle.HIGHEST_PROTOCOL)
            w_file.write(binary_spec)
            spec_list.append((line_idx, len(spec), offset, len(binary_spec)))
            offset += len(binary_spec)

            if not spec_max or spec_max < spec.max():
                spec_max = spec.max()

            if not spec_min or spec_min < spec.min():
                spec_min = spec.min()

    print(f_type, 'spectrogram max/min', spec_max, spec_min)

    q.put((f_type, spec_list))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='training script')
        parser.add_argument('--dataset', type=str, default='10', help='vctk / blizzard / 10 / nancy / etri')
        args = parser.parse_args()
        dataset = args.dataset
        print('Dataset to preprocess:', dataset)
        write_csv = True

        if dataset == 'blizzard':
            dir_text = '/data2/lyg0722/TTS_corpus/blizzard/segmented/txt/'
            dir_spec = '/data2/lyg0722/TTS_corpus/blizzard/segmented/wav/'
            dir_bin = '/data2/lyg0722/TTS_corpus/blizzard/segmented/bin/'
        elif dataset == 'etri':
            dir_text = '/data2/lyg0722/TTS_corpus/etri/txt/'
            dir_spec = '/data2/lyg0722/TTS_corpus/etri/wav/'
            dir_bin = '/data2/lyg0722/TTS_corpus/etri/bin/'

        q = Queue()
        p_lin = Process(target=preprocess_spec, args=(dataset, 'linear', dir_spec, dir_bin, q))
        p_mel = Process(target=preprocess_spec, args=(dataset, 'mel', dir_spec, dir_bin, q))
        p_lin.daemon = True
        p_mel.daemon = True
        p_lin.start()
        p_mel.start()

        lin_list = None
        tmp_get = q.get()

        if tmp_get[0] == 'mel':
            mel_list = tmp_get[1]
        else:
            lin_list = tmp_get[1]
            mel_list = q.get()[1]

        # text part
        get_vocab(dataset, dir_text)
        txt_list = cleanse(dir_text, dir_bin, write_csv, mel_list)

        if not lin_list:
            lin_list = q.get()[1]

        p_lin.join()
        p_mel.join()

        # make file load list
        assert len(txt_list) == len(mel_list)
        line_load_list = []
        for i, item in enumerate(mel_list):
            assert item[0] == txt_list[i][0] and item[0] == lin_list[i][0]
            line_idx = item[0]
            wave_length = item[1]
            text_length = txt_list[i][1]
            offset_M = item[2]
            offset_L = lin_list[i][2]
            offset_T = txt_list[i][2]
            len_M = item[3]
            len_L = lin_list[i][3]
            len_T = txt_list[i][3]
            line_load_list.append((i, wave_length, text_length, offset_M, offset_L, offset_T, len_M, len_L, len_T))

        torch.save(torch.DoubleTensor(line_load_list), dir_bin + 'line_load_list.t7')
    finally:
        for p in multiprocessing.active_children():
            p.terminate()
