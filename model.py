# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random


class Tacotron(nn.Module):
    def __init__(self, args):
        super(Tacotron, self).__init__()
        self.trunc_size = args.trunc_size
        self.r_factor = args.r_factor
        self.dec_out_size = args.dec_out_size

        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.attn_weights = []      # only used in evaluation

        self.encoder = Encoder(args.vocab_size, args.charvec_dim, args.hidden_size, args.num_filters, args.dropout)
        self.linear_enc = nn.Linear(2 * args.hidden_size, 2 * args.hidden_size, bias=False)                          # N*T_enc x 2H

        self.decoder = AttnDecoderRNN(args.hidden_size, args.dec_out_size, args.r_factor, args.dropout)
        self.post_processor = PostProcessor(args.hidden_size, args.dec_out_size, args.post_out_size, args.num_filters // 2)


    def forward(self, enc_input, dec_input, wave_lengths, text_lengths, prev_h):
        r = self.r_factor
        T_wav, T_dec = max(wave_lengths), max(wave_lengths)//r

        enc_output = self.encoder(enc_input, text_lengths)
        in_attW_enc = rnn.pack_padded_sequence(enc_output, text_lengths, True)
        in_attW_enc = self.linear_enc(in_attW_enc.data)                             # N*T_enc x 2H

        output_mel_list = []
        prev_dec_output = dec_input[:, 0]
        h_att, h_dec1, h_dec2 = prev_h

        for di in range(T_dec):
            start_idx, end_idx = di*r, (di+1)*r

            prev_dec_output, h_att, h_dec1, h_dec2 = self.decoder(
                enc_output, in_attW_enc, prev_dec_output, text_lengths, h_att, h_dec1, h_dec2)

            output_mel_list.append(prev_dec_output)

            if random.random() < self.teacher_forcing_ratio:
                prev_dec_output = dec_input[:, end_idx-1]                       # Teacher forcing
            else:
                prev_dec_output = output_mel_list[-1][:, -1]

            if not self.training:
                self.attn_weights.append(self.decoder.attn_weights.data)

            # TODO: make it stop when it meets EOS token

        output_mel = torch.cat(output_mel_list, dim=1)
        output_linear = self.post_processor(output_mel)
        last_h = (Variable(h_att.data), Variable(h_dec1.data), Variable(h_dec2.data))
        return output_mel, output_linear, last_h


class Encoder(nn.Module):
    """ input[0]: NxT sized Tensor
        input[1]: B sized lengths Tensor
        output: NxTxH sized Tensor
    """
    def __init__(self, vocab_size, charvec_dim, hidden_size, num_filters, dropout_p=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, charvec_dim)
        self.prenet = nn.Sequential(
            nn.Linear(charvec_dim, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.CBHG = CBHG(hidden_size, hidden_size, hidden_size, hidden_size, hidden_size, num_filters, True)

    def forward(self, input, lengths):
        N, T = input.size(0), input.size(1)
        embedded = self.embedding(input).view(N*T, -1)          # NT x C
        output = self.prenet(embedded).view(N, T, -1)           # N x T x H
        output = self.CBHG(output, lengths)
        return output


class AttnDecoderRNN(nn.Module):
    """ input_enc: Output from encoder (NxTx2H)
        input_attW_enc: masked-linear transformed input_enc
        input_dec: Output from previous-step decoder (NxO_dec)
        lengths: N sized Tensor
        output: N x r x H sized Tensor
    """
    def __init__(self, hidden_size, output_size, r_factor=2, dropout_p=0.5):
        super(AttnDecoderRNN, self).__init__()
        self.r_factor = r_factor

        self.prenet = nn.Sequential(
            nn.Linear(output_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.linear_dec = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.gru_att = nn.GRU(hidden_size, 2 * hidden_size, batch_first=True)

        self.attn = nn.Linear(2 * hidden_size, 1)       # TODO: change name...

        self.short_cut = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.gru_dec1 = nn.GRU(4 * hidden_size, 2 * hidden_size, num_layers=1, batch_first=True)
        self.gru_dec2 = nn.GRU(2 * hidden_size, 2 * hidden_size, num_layers=1, batch_first=True)

        self.out = nn.Linear(2 * hidden_size, r_factor * output_size)

    def forward(self, input_enc, input_attW_enc, input_dec, lengths_enc, hidden_att=None, hidden_dec1=None, hidden_dec2=None):
        N = input_dec.size(0)

        out_att = self.prenet(input_dec).unsqueeze(1)                                   # N x O_dec -> N x 1 x H
        out_att, hidden_att = self.gru_att(out_att, hidden_att)                         # N x 1 x 2H
        in_attW_dec = self.linear_dec(out_att.squeeze(1)).unsqueeze(1).expand_as(input_enc)
        in_attW_dec = rnn.pack_padded_sequence(in_attW_dec, lengths_enc, True)          # N*T_enc x 2H

        self.attn_weights = torch.add(input_attW_enc, in_attW_dec.data).tanh()          # N x T_enc x 2H
        self.attn_weights = self.attn(self.attn_weights).exp()                          # N*T_enc x 1
        self.attn_weights = rnn.PackedSequence(self.attn_weights, in_attW_dec.batch_sizes)
        self.attn_weights, _ = rnn.pad_packed_sequence(self.attn_weights, True)
        self.attn_weights = F.normalize(self.attn_weights, 1, 1)                        # N x T_enc x 1

        attn_applied = torch.bmm(self.attn_weights.transpose(1,2), input_enc)           # N x 1 x 2H

        out_dec = torch.cat((attn_applied, out_att), 2)                                 # N x 1 x 4H
        residual = self.short_cut(out_dec.squeeze(1)).unsqueeze(1)                      # N x 1 x 2H

        out_dec, hidden_dec1 = self.gru_dec1(out_dec, hidden_dec1)
        residual = residual + out_dec

        out_dec, hidden_dec2 = self.gru_dec2(residual, hidden_dec2)
        residual = residual + out_dec

        output = self.out(residual.squeeze(1)).view(N, self.r_factor, -1)
        return output, hidden_att, hidden_dec1, hidden_dec2


class PostProcessor(nn.Module):
    """ input: N x T x O_dec
        output: N x T x O_post
    """
    def __init__(self, hidden_size, dec_out_size, post_out_size, num_filters):
        super(PostProcessor, self).__init__()
        self.CBHG = CBHG(dec_out_size, hidden_size, 2 * hidden_size, hidden_size, hidden_size, num_filters, True)
        self.projection = nn.Linear(2 * hidden_size, post_out_size)

    def forward(self, input, lengths=None):
        if lengths is None:
            N, T = input.size(0), input.size(1)
            lengths = [T for _ in range(N)]
            output = self.CBHG(input, lengths).contiguous().view(N*T,-1)
            output = self.projection(output).view(N,T,-1)
        else:
            output = self.CBHG(input, lengths)
            output = rnn.pack_padded_sequence(output, lengths, True)
            output = rnn.PackedSequence(self.projection(output.data), output.batch_sizes)
            output, _ = rnn.pad_packed_sequence(output, True)
        return output


class CBHG(nn.Module):
    """ input: NxTxinput_dim sized Tensor
        output: NxTx2gru_dim sized Tensor
    """
    def __init__(self, input_dim, conv_bank_dim, conv_dim1, conv_dim2, gru_dim, num_filters, is_masked):
        super(CBHG, self).__init__()
        self.num_filters = num_filters

        bank_out_dim = num_filters * conv_bank_dim
        self.conv_bank = nn.ModuleList()
        for i in range(num_filters):
            self.conv_bank.append(nn.Conv1d(input_dim, conv_bank_dim, i + 1, stride=1, padding=int(np.ceil(i / 2))))

        # define batch normalization layer, we use BN1D since the sequence length is not fixed
        self.bn_list = nn.ModuleList()
        self.bn_list.append(nn.BatchNorm1d(bank_out_dim))
        self.bn_list.append(nn.BatchNorm1d(conv_dim1))
        self.bn_list.append(nn.BatchNorm1d(conv_dim2))

        self.conv1 = nn.Conv1d(bank_out_dim, conv_dim1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv_dim1, conv_dim2, 3, stride=1, padding=1)

        if input_dim != conv_dim2:
            self.residual_proj = nn.Linear(input_dim, conv_dim2)

        self.highway = Highway(conv_dim2, 4)
        self.BGRU = nn.GRU(input_size=conv_dim2, hidden_size=gru_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input, lengths):
        N, T = input.size(0), input.size(1)

        conv_bank_out = []
        input_t = input.transpose(1, 2)  # NxTxH -> NxHxT
        for i in range(self.num_filters):
            tmp_input = input_t
            if i % 2 == 0:
                tmp_input = tmp_input.unsqueeze(-1)
                tmp_input = F.pad(tmp_input, (0,0,0,1)).squeeze(-1)   # NxHxT
            conv_bank_out.append(self.conv_bank[i](tmp_input))

        residual = torch.cat(conv_bank_out, dim=1)                  # NxHFxT
        residual = F.relu(self.bn_list[0](residual))
        residual = F.max_pool1d(residual, 2, stride=1)
        residual = self.conv1(residual)                             # NxHxT
        residual = F.relu(self.bn_list[1](residual))
        residual = self.conv2(residual)                             # NxHxT
        residual = self.bn_list[2](residual).transpose(1,2)         # NxHxT -> NxTxH

        rnn_input = input
        if rnn_input.size() != residual.size():
            rnn_input = self.residual_proj(rnn_input)
        rnn_input = rnn_input + residual
        rnn_input = self.highway(rnn_input).view(N, T, -1)

        output = rnn.pack_padded_sequence(rnn_input, lengths, True)
        output, _ = self.BGRU(output)                               # zero h_0 is used by default
        output, _ = rnn.pad_packed_sequence(output, True)           # NxTx2H
        return output


class Highway(nn.Module):
    """
    Code from: https://github.com/kefirski/pytorch_Highway
    """
    def __init__(self, size, num_layers, f=F.relu):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """ input: NxH sized Tensor
            output: NxH sized Tensor
        """
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x
