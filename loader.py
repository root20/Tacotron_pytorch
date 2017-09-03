# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from torch.multiprocessing import Process, Queue, Pool
from torch.autograd import Variable
from functools import partial
import math, torch, pickle
import os.path


class DataLoader():
    def __init__(self, args):
        self.dir_bin = args.dir_bin
        line_load_list = self.dir_bin + 'line_load_list.t7'
        vocab_file = self.dir_bin + 'vocab.t7'
        assert os.path.isfile(self.dir_bin + 'specM.bin')
        assert os.path.isfile(self.dir_bin + 'specL.bin')
        assert os.path.isfile(self.dir_bin + 'text.bin')

        self.batch_size = args.batch_size
        self.trunc_size = args.trunc_size
        self.r_factor = args.r_factor
        self.dec_out_size = args.dec_out_size
        self.post_out_size = args.post_out_size
        self.shuffle_data = True if args.shuffle_data == 1 else False
        self.iter_per_epoch = None
        self.is_subbatch_end = True
        self.curr_split = None
        self.vocab_size = None

        self.process = None
        self.queue = Queue(maxsize=args.load_queue_size)
        self.n_workers = args.n_workers

        self.use_gpu = args.use_gpu
        self.num_gpu = len(args.gpu) if len(args.gpu) > 0 else 1
        self.pinned_memory = True if args.pinned_memory == 1 and self.use_gpu else False

        self.vocab_size = self.get_num_vocab(vocab_file)
        text_limit = args.text_limit
        wave_limit = args.wave_limit

        # col1: idx / col2: wave_length / col3: text_length
        # col4: offset_M / col5: offset_L / col6: offset_T
        self.load_list = torch.load(line_load_list)
        spec_len_list = self.load_list[:, 1].clone()
        text_len_list = self.load_list[:, 2].clone()

        # exclude files whose wave length exceeds wave_limit
        sort_length, sort_idx = spec_len_list.sort()
        text_len_list = torch.gather(text_len_list, 0, sort_idx)
        sort_idx = sort_idx.view(-1, 1).expand_as(self.load_list)
        self.load_list = torch.gather(self.load_list, 0, sort_idx)

        end_idx = sort_length.le(wave_limit).sum()
        spec_len_list = sort_length[:end_idx]
        text_len_list = text_len_list[:end_idx]
        self.load_list = self.load_list[:end_idx]

        # exclude files whose text length exceeds text_limit
        sort_length, sort_idx = text_len_list.sort()
        spec_len_list = torch.gather(spec_len_list, 0, sort_idx)
        sort_idx = sort_idx.view(-1, 1).expand_as(self.load_list)
        self.load_list = torch.gather(self.load_list, 0, sort_idx)

        end_idx = sort_length.le(text_limit).sum()
        end_idx = end_idx - (end_idx % self.batch_size)  # drop residual data
        text_len_list = sort_length[:end_idx]
        spec_len_list = spec_len_list[:end_idx]
        self.load_list = self.load_list[:end_idx]

        # sort by wave length
        _, sort_idx = spec_len_list.sort(0, descending=True)
        text_len_list = torch.gather(text_len_list, 0, sort_idx)
        sort_idx = sort_idx.view(-1, 1).expand_as(self.load_list)
        self.load_list = torch.gather(self.load_list, 0, sort_idx)

        # sort by text length in each batch (PackedSequence requires it)
        num_batches_per_epoch = self.load_list.size(0) // self.batch_size
        text_len_list = text_len_list.view(num_batches_per_epoch, -1)
        self.load_list = self.load_list.view(num_batches_per_epoch, -1, self.load_list.size(1))
        sort_length, sort_idx = text_len_list.sort(1, descending=True)
        sort_idx = sort_idx.view(num_batches_per_epoch, -1, 1).expand_as(self.load_list)
        self.load_list = torch.gather(self.load_list, 1, sort_idx)

        # shuffle while preserving order in a batch
        if self.shuffle_data:
            _, sort_idx = torch.randn(num_batches_per_epoch).sort()
            sort_idx = sort_idx.view(-1, 1, 1).expand_as(self.load_list)
            self.load_list = torch.gather(self.load_list, 0, sort_idx)      # nbpe x N x 6

        self.load_list = self.load_list.long()

        # compute number of iterations needed
        spec_len_list = spec_len_list.view(num_batches_per_epoch, -1)
        spec_len_list, _ = spec_len_list.div(self.trunc_size).ceil().max(1)
        self.iter_per_epoch = int(spec_len_list.sum())

        # set split cursor
        self.split_sizes = {'train': self.load_list.size(0), 'val': -1, 'test': -1}
        self.split_cursor = {'train': 0, 'val': 0, 'test': 0}


    def next_batch(self, split):
        T, idx = self.trunc_size, self.split_cursor[split]

        # seek and load data from raw files
        if self.is_subbatch_end:
            self.is_subbatch_end = False
            self.subbatch_cursor = 0

            if self.curr_split != split:
                self.curr_split = split
                if self.process is not None:
                    self.process.terminate()
                self.process = Process(target=self.start_async_loader, args=(split, self.split_cursor[split]))
                self.process.start()

            self.len_text, self.len_wave, self.curr_text, self.curr_specM, self.curr_specL = self.queue.get()
            self.split_cursor[split] = (idx + 1) % self.split_sizes[split]
            self.subbatch_max_len = self.len_wave.max()

        # Variables to return
        # +1 to length of y to consider shifting for target y
        subbatch_len_text = [x for x in self.len_text]
        subbatch_len_wave = [min(x, T) for x in self.len_wave]
        x_text = self.curr_text
        y_specM = self.curr_specM[:, self.subbatch_cursor:self.subbatch_cursor + max(subbatch_len_wave) + 1].contiguous()
        y_specL = self.curr_specL[:, self.subbatch_cursor:self.subbatch_cursor + max(subbatch_len_wave) + 1].contiguous()

        if self.use_gpu:
            if self.pinned_memory:
                x_text = x_text.pin_memory()
                y_specM = y_specM.pin_memory()
                y_specL = y_specL.pin_memory()

            x_text = x_text.cuda()
            y_specM = y_specM.cuda()
            y_specL = y_specL.cuda()

        # Advance split_cursor or Move on to the next batch
        if self.subbatch_cursor + T < self.subbatch_max_len:
            self.subbatch_cursor = self.subbatch_cursor + T
            self.len_wave.sub_(T).clamp_(min=0)
        else:
            self.is_subbatch_end = True

        # Don't compute for empty batch elements
        if subbatch_len_wave.count(0) > 0:
            self.len_wave_mask = [idx for idx, l in enumerate(subbatch_len_wave) if l > 0]
            self.len_wave_mask = torch.LongTensor(self.len_wave_mask)
            if self.use_gpu:
                self.len_wave_mask = self.len_wave_mask.cuda()

            x_text = torch.index_select(x_text, 0, self.len_wave_mask)
            y_specM = torch.index_select(y_specM, 0, self.len_wave_mask)
            y_specL = torch.index_select(y_specL, 0, self.len_wave_mask)
            subbatch_len_text = [subbatch_len_text[idx] for idx in self.len_wave_mask]
            subbatch_len_wave = [subbatch_len_wave[idx] for idx in self.len_wave_mask]
        else:
            self.len_wave_mask = None

        return x_text, y_specM, y_specL, subbatch_len_wave, subbatch_len_text


    def start_async_loader(self, split, load_start_idx):
        # load batches to the queue asynchronously since it is a bottle-neck
        N, r = self.batch_size, self.r_factor
        load_curr_idx = load_start_idx

        while True:
            data_T, data_M, data_L, len_T, len_M = ([None for _ in range(N)] for _ in range(5))
            # deploy workers to load data
            self.pool = Pool(self.n_workers)
            partial_func = partial(load_data_and_length, self.dir_bin, self.load_list[load_curr_idx])
            results = self.pool.map_async(func=partial_func, iterable=range(N))
            self.pool.close()
            self.pool.join()

            for result in results.get():
                data_M[result[0]] = result[1]
                data_L[result[0]] = result[2]
                data_T[result[0]] = result[3]
                len_T[result[0]] = result[4]
                len_M[result[0]] = result[5]

            # TODO: output size is not accurate.. //
            len_text = torch.IntTensor(len_T)
            len_wave = torch.Tensor(len_M).div(r).ceil().mul(r).int()                       # consider r_factor
            curr_text = torch.LongTensor(N, len_text.max()).fill_(0)                        # null-padding at tail
            curr_specM = torch.Tensor(N, len_wave.max() + 1, self.dec_out_size).fill_(0)    # null-padding at tail
            curr_specL = torch.Tensor(N, len_wave.max() + 1, self.post_out_size).fill_(0)   # null-padding at tail

            # fill the template tensors
            for j in range(N):
                curr_text[j, 0:data_T[j].size(0)].copy_(data_T[j])
                curr_specM[j, 0:data_M[j].size(0)].copy_(data_M[j])
                curr_specL[j, 0:data_L[j].size(0)].copy_(data_L[j])

            self.queue.put((len_text, len_wave, curr_text, curr_specM, curr_specL))
            load_curr_idx = (load_curr_idx + 1) % self.split_sizes[split]


    def mask_prev_h(self, prev_h):
        if self.len_wave_mask is not None:
            if self.use_gpu:
                self.len_wave_mask = self.len_wave_mask.cuda()

            h_att, h_dec1, h_dec2 = prev_h
            h_att = torch.index_select(h_att.data, 1, self.len_wave_mask)  # batch idx is
            h_dec1 = torch.index_select(h_dec1.data, 1, self.len_wave_mask)
            h_dec2 = torch.index_select(h_dec2.data, 1, self.len_wave_mask)
            prev_h = (Variable(h_att), Variable(h_dec1), Variable(h_dec2))
        else:
            prev_h = prev_h

        return prev_h


    def get_num_vocab(self, vocab_file=None):
        if self.vocab_size:
            return self.vocab_size
        else:
            vocab_dict = torch.load(vocab_file)
            return len(vocab_dict) + 1  # +1 to consider null-padding


def load_binary(file_path, offset, length):
    with open(file_path, 'rb') as datafile:
        datafile.seek(offset)
        line = datafile.read(length)
        obj = pickle.loads(line)
    return obj


def load_data_and_length(dir_bin, load_info, load_idx):
    # If out of range error occurs at here, check whether you are using right text_limit.
    data_M = load_binary(dir_bin + 'specM.bin', load_info[load_idx][3], load_info[load_idx][6])
    data_L = load_binary(dir_bin + 'specL.bin', load_info[load_idx][4], load_info[load_idx][7])
    data_T = load_binary(dir_bin + 'text.bin', load_info[load_idx][5], load_info[load_idx][8])
    # Convert to Tensor
    data_M = torch.from_numpy(data_M)
    data_L = torch.from_numpy(data_L)
    data_T = torch.LongTensor(data_T)

    len_M = data_M.size(0)
    len_T = data_T.size(0)
    return (load_idx, data_M, data_L, data_T, len_T, len_M)

