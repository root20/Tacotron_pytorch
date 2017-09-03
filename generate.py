# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import argparse, librosa, copy, shutil, pdb, multiprocessing, re
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import Tacotron as Tacotron
from loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='training script')
    # data load
    parser.add_argument('--data', type=str, default='blizzard', help='blizzard / nancy')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--text_limit', type=int, default=1500, help='maximum length of text to include in training set')
    parser.add_argument('--wave_limit', type=int, default=800, help='maximum length of spectrogram to include in training set')
    parser.add_argument('--shuffle_data', type=int, default=0, help='whether to shuffle data loader')
    parser.add_argument('--batch_idx', type=int, default=0, help='n-th batch of the dataset')
    parser.add_argument('--load_queue_size', type=int, default=1, help='maximum number of batches to load on the memory')
    parser.add_argument('--n_workers', type=int, default=1, help='number of workers used in data loader')
    # generation option
    parser.add_argument('--exp_no', type=int, default=0, help='')
    parser.add_argument('--out_dir', type=str, default='generated', help='')
    parser.add_argument('--init_from', type=str, default='', help='load parameters from...')
    parser.add_argument('--caption', type=str, default='', help='text to generate speech')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0, help='value between 0~1, use this for scheduled sampling')
    # audio related option
    parser.add_argument('--n_fft', type=int, default=2048, help='fft bin size')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sampling rate')
    parser.add_argument('--frame_len_inMS', type=int, default=50, help='used to determine window size of fft')
    parser.add_argument('--frame_shift_inMS', type=int, default=12.5, help='used to determine stride in sfft')
    parser.add_argument('--num_recon_iters', type=int, default=50, help='# of iteration in griffin-lim recon')
    # misc
    parser.add_argument('--gpu', type=int, nargs='+', help='index of gpu machines to run')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    new_args = vars(parser.parse_args())

    # load and override some arguments
    checkpoint = torch.load(new_args['init_from'], map_location=lambda storage, loc: storage)
    args = checkpoint['args']
    for i in new_args:
        args.__dict__[i] = new_args[i]

    torch.manual_seed(args.seed)

    # set dataset option
    if args.data == 'blizzard':
        args.dir_bin = '/data2/lyg0722/TTS_corpus/blizzard/segmented/bin/'
    elif args.data == 'etri':
        args.dir_bin = '/data2/lyg0722/TTS_corpus/etri/bin/'
    else:
        print('no dataset')
        return

    if args.gpu is None:
        args.use_gpu = False
        args.gpu = []
    else:
        args.use_gpu = True
        torch.cuda.manual_seed(0)
        torch.cuda.set_device(args.gpu[0])

    model = Tacotron(args)
    criterion_mel = nn.L1Loss(size_average=False)
    criterion_lin = nn.L1Loss(size_average=False)

    window_len = int(np.ceil(args.frame_len_inMS * args.sample_rate / 1000))
    hop_length = int(np.ceil(args.frame_shift_inMS * args.sample_rate / 1000))

    if args.init_from:
        model.load_state_dict(checkpoint['state_dict'])
        print('loaded checkpoint %s' % (args.init_from))

    model = model.eval()

    if args.use_gpu:
        model = model.cuda()
        criterion_mel = criterion_mel.cuda()
        criterion_lin = criterion_lin.cuda()

    if args.caption:
        text_raw = args.caption

        if args.data == 'etri':
            text_raw = decompose_hangul(text_raw)       # For Korean dataset

        vocab_dict = torch.load(args.dir_bin + 'vocab.t7')

        enc_input = [vocab_dict[i] for i in text_raw]
        enc_input = enc_input + [0]                                   # null-padding at tail
        text_lengths = [len(enc_input)]
        enc_input = Variable(torch.LongTensor(enc_input).view(1,-1))

        dec_input = torch.Tensor(1, 1, args.dec_out_size).fill_(0)          # null-padding for start flag
        dec_input = Variable(dec_input)
        wave_lengths = [args.wave_limit]        # TODO: use <EOS> later...

        prev_h = (None, None, None)  # set prev_h = h_0 when new sentences are loaded

        if args.gpu:
            enc_input = enc_input.cuda()
            dec_input = dec_input.cuda()

        _, pred_lin, prev_h = model(enc_input, dec_input, wave_lengths, text_lengths, prev_h)

        # start generation
        wave = spectrogram2wav(
            pred_lin.data.view(-1, args.post_out_size).cpu().numpy(),
            n_fft=args.n_fft,
            win_length=window_len,
            hop_length=hop_length,
            num_iters=args.num_recon_iters
        )

        # write to file
        outpath1 = '%s/%s_%s.wav' % (args.out_dir, args.exp_no, args.caption)
        outpath2 = '%s/%s_%s.png' % (args.out_dir, args.exp_no, args.caption)
        librosa.output.write_wav(outpath1, wave, 16000)
        saveAttention(text_raw, torch.cat(model.attn_weights, dim=-1).squeeze(), outpath2)
    else:
        loader = DataLoader(args)
        args.vocab_size = loader.get_num_vocab()

        for iter in range(1, loader.iter_per_epoch + 1):
            if loader.is_subbatch_end:
                prev_h = (None, None, None)  # set prev_h = h_0 when new sentences are loaded

            for i in range(args.batch_idx):
                loader.next_batch('train')

            enc_input, target_mel, target_lin, wave_lengths, text_lengths = loader.next_batch('train')
            enc_input = Variable(enc_input, volatile=True)
            target_mel = Variable(target_mel, volatile=True)
            target_lin = Variable(target_lin, volatile=True)

            prev_h = loader.mask_prev_h(prev_h)

            if args.gpu:
                enc_input = enc_input.cuda()
                target_mel = target_mel.cuda()
                target_lin = target_lin.cuda()

            pred_mel, pred_lin, prev_h = model(enc_input, target_mel[:, :-1], wave_lengths, text_lengths, prev_h)

            loss_mel = criterion_mel(pred_mel, target_mel[:, 1:]) \
                .div(max(wave_lengths) * args.batch_size * args.dec_out_size)
            loss_linear = criterion_lin(pred_lin, target_lin[:, 1:]) \
                .div(max(wave_lengths) * args.batch_size * args.post_out_size)
            loss = torch.sum(loss_mel + loss_linear)

            print('loss:' , loss.data[0])

            attentions = torch.cat(model.attn_weights, dim=-1)

            # write to file
            for n in range(enc_input.size(0)):
                wave = spectrogram2wav(
                    pred_lin.data[n].view(-1, args.post_out_size).cpu().numpy(),
                    n_fft=args.n_fft,
                    win_length=window_len,
                    hop_length=hop_length,
                    num_iters=args.num_recon_iters
                )
                outpath1 = '%s/%s_%s_%s.wav' % (args.out_dir, args.exp_no, n, args.caption)
                librosa.output.write_wav(outpath1, wave, 16000)
                outpath2 = '%s/%s_%s_%s.png' % (args.out_dir, args.exp_no, n, args.caption)
                saveAttention(None, attentions[n], outpath2)


            # showPlot(plot_losses)
            break

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def saveAttention(input_sentence, attentions, outpath):
    # Set up figure with colorbar
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig = plt.figure(figsize=(24,10), )
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    if input_sentence:
        # Set up axes
        ax.set_yticklabels([' '] + list(input_sentence) + [' '])
        # Show label at every tick
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')


def spectrogram2wav(spectrogram, n_fft, win_length, hop_length, num_iters):
    '''
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    min_level_db = -100
    ref_level_db = 20

    spec = spectrogram.T
    # denormalize
    spec = (np.clip(spec, 0, 1) * - min_level_db) + min_level_db
    spec = spec + ref_level_db

    # Convert back to linear
    spec = np.power(10.0, spec * 0.05)

    return _griffin_lim(spec ** 1.5, n_fft, win_length, hop_length, num_iters)  # Reconstruct phase


def _griffin_lim(S, n_fft, win_length, hop_length, num_iters):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    for i in range(num_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)))
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return y


def decompose_hangul(text):
    """
    Code from: https://github.com/neotune/python-korean-handler
    """

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

    line_dec = ""
    line = list(text.strip())

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

    return line_dec

if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            p.terminate()
