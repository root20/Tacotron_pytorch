# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import argparse, multiprocessing

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from model import Tacotron as Tacotron
from loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='training script')
    # data load
    parser.add_argument('--data', type=str, default='blizzard', help='blizzard / nancy')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--text_limit', type=int, default=1000, help='maximum length of text to include in training set')
    parser.add_argument('--wave_limit', type=int, default=1400, help='maximum length of spectrogram to include in training set')
    parser.add_argument('--trunc_size', type=int, default=700, help='used for truncated-BPTT when memory is not enough.')
    parser.add_argument('--shuffle_data', type=int, default=1, help='whether to shuffle data loader')
    parser.add_argument('--load_queue_size', type=int, default=8, help='maximum number of batches to load on the memory')
    parser.add_argument('--n_workers', type=int, default=2, help='number of workers used in data loader')
    # model
    parser.add_argument('--charvec_dim', type=int, default=256, help='')
    parser.add_argument('--hidden_size', type=int, default=128, help='')
    parser.add_argument('--dec_out_size', type=int, default=80, help='decoder output size')
    parser.add_argument('--post_out_size', type=int, default=1025, help='should be n_fft / 2 + 1(check n_fft from "input_specL" ')
    parser.add_argument('--num_filters', type=int, default=16, help='number of filters in filter bank of CBHG')
    parser.add_argument('--r_factor', type=int, default=5, help='reduction factor(# of multiple output)')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    # optimization
    parser.add_argument('--max_epochs', type=int, default=100000, help='maximum epoch to train')
    parser.add_argument('--grad_clip', type=float, default=1, help='gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='2e-3 from Ito, I used to use 5e-4')
    parser.add_argument('--lr_decay_every', type=int, default=25000, help='decay learning rate every...')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='decay learning rate by this factor')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1, help='value between 0~1, use this for scheduled sampling')
    # loading
    parser.add_argument('--init_from', type=str, default='', help='load parameters from...')
    parser.add_argument('--resume', type=int, default=0, help='1 for resume from saved epoch')
    # misc
    parser.add_argument('--exp_no', type=int, default=0, help='')
    parser.add_argument('--print_every', type=int, default=-1, help='')
    parser.add_argument('--plot_every', type=int, default=-1, help='')
    parser.add_argument('--save_every', type=int, default=-1, help='')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='')
    parser.add_argument('--pinned_memory', type=int, default=1, help='1 to use pinned memory')
    parser.add_argument('--gpu', type=int, nargs='+', help='index of gpu machines to run')
    # debug
    parser.add_argument('--debug', type=int, default=0, help='1 for debug mode')
    args = parser.parse_args()

    torch.manual_seed(0)

    # set dataset option
    if args.data == 'blizzard':
        args.dir_bin = '/home/lyg0722/TTS_corpus/blizzard/segmented/bin/'
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

    loader = DataLoader(args)

    # set misc options
    args.vocab_size = loader.get_num_vocab()
    if args.print_every == -1:
        args.print_every = loader.iter_per_epoch
    if args.plot_every == -1:
        args.plot_every = args.print_every
    if args.save_every == -1:
        args.save_every = loader.iter_per_epoch * 10    # save every 10 epoch by default

    model = Tacotron(args)
    model_optim = optim.Adam(model.parameters(), args.learning_rate)
    criterion_mel = nn.L1Loss(size_average=False)
    criterion_lin = nn.L1Loss(size_average=False)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    start_epoch = 0
    iter = 1

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if args.resume != 0:
            start_epoch = checkpoint['epoch']
            plot_losses = checkpoint['plot_losses']
        print('loaded checkpoint %s (epoch %d)' % (args.init_from, start_epoch))

    model = model.train()
    if args.use_gpu:
        model = model.cuda()
        criterion_mel = criterion_mel.cuda()
        criterion_lin = criterion_lin.cuda()

    print('Start training... (1 epoch = %s iters)' % (loader.iter_per_epoch))
    while iter < args.max_epochs * loader.iter_per_epoch + 1:
        if loader.is_subbatch_end:
            prev_h = (None, None, None)             # set prev_h = h_0 when new sentences are loaded
        enc_input, target_mel, target_lin, wave_lengths, text_lengths = loader.next_batch('train')

        max_wave_len = max(wave_lengths)

        enc_input = Variable(enc_input, requires_grad=False)
        target_mel = Variable(target_mel, requires_grad=False)
        target_lin = Variable(target_lin, requires_grad=False)

        prev_h = loader.mask_prev_h(prev_h)

        model_optim.zero_grad()
        pred_mel, pred_lin, prev_h = model(enc_input, target_mel[:, :-1], wave_lengths, text_lengths, prev_h)

        loss_mel = criterion_mel(pred_mel, target_mel[:, 1:])\
                        .div(max_wave_len * args.batch_size * args.dec_out_size)
        loss_linear = criterion_lin(pred_lin, target_lin[:, 1:])\
                        .div(max_wave_len * args.batch_size * args.post_out_size)
        loss = torch.sum(loss_mel + loss_linear)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)         # gradient clipping
        model_optim.step()

        print_loss_total += loss.data[0]
        plot_loss_total += loss.data[0]

        if iter % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / args.max_epochs),
                                         iter, iter / args.max_epochs * 100, print_loss_avg))
        if iter % args.plot_every == 0:
            plot_loss_avg = plot_loss_total / args.plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            save_name = '%s/%dth_exp_loss.png' % (args.save_dir, args.exp_no)
            savePlot(plot_losses, save_name)


        if iter % args.save_every == 0:
            epoch = start_epoch + iter // loader.iter_per_epoch
            save_name = '%s/%d_%dth.t7' % (args.save_dir, args.exp_no, epoch)
            state = {
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': model_optim.state_dict(),
                'plot_losses': plot_losses
            }
            torch.save(state, save_name)
            print('model saved to', save_name)
            # if is_best:               # TODO: implement saving best model.
            #     shutil.copyfile(save_name, '%s/%d_best.t7' % (args.save_dir, args.exp_no))

        iter += 1


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

def savePlot(points, outpath):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(outpath)
    plt.close('all')

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time, math
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


if __name__ == '__main__':
    try:
        main()
    finally:
        for p in multiprocessing.active_children():
            # p.join()
            p.terminate()
