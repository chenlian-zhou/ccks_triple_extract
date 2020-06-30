# coding: utf8

import torch
import torch.nn as nn

from data_utils import START_TAG_IDX, STOP_TAG_IDX, PAD_IDX
from utils import scalar, LongTensor, Tensor, zeros, randn

WORD_EMBEDDING_DIM = 300
WORD_LSTM_HIDDEN_SIZE = 600
WORD_LSTM_NUM_LAYERS = 1
WORD_LSTM_BIDIRECTIONAL = True
WORD_LSTM_NUM_DIRS = 2 if WORD_LSTM_BIDIRECTIONAL else 1

CHAR_EMBEDDING_DIM = 100
CHAR_LSTM_HIDDEN_SIZE = 200
CHAR_LSTM_NUM_LAYERS = 1
CHAR_LSTM_BIDIRECTIONAL = True
CHAR_LSTM_NUM_DIRS = 2 if CHAR_LSTM_BIDIRECTIONAL else 1


def _sort(_2dtensor, lengths, descending=True):
    sorted_lengths, order = lengths.sort(descending=descending)
    _2dtensor_sorted_by_lengths = _2dtensor[order]
    return _2dtensor_sorted_by_lengths, order


class CRFOnLSTM(nn.Module):
    def __init__(
            self,
            num_word_embeddings,
            num_tags,
            word_embeddings,
            num_char_embeddings,
            word_lstm,
            char_lstm):
        super(CRFOnLSTM, self).__init__()
        self.lstm = WordCharLSTM(
            num_word_embeddings,
            num_tags,
            word_embeddings,
            num_char_embeddings,
            word_lstm=word_lstm,
            char_lstm=char_lstm)

        self.crf = CRF(num_tags)
        self = self.cuda() if torch.cuda.is_available() else self

    def forward(self, word_x, char_x, y):  # for training
        mask = word_x.data.gt(0).float()  # because 0 is pad_idx, doesn't really belong here, I guess
        h = self.lstm(word_x, mask, char_x)
        Z = self.crf.forward(h, mask)  # partition function
        score = self.crf.score(h, y, mask)
        return Z - score  # NLL loss

    def decode(self, word_x, char_x):  # for prediction
        mask = word_x.data.gt(0).float()  # again 0 is probably because of pad_idx, maybe pass mask as parameter
        h = self.lstm(word_x, mask, char_x)
        return self.crf.decode(h, mask)


class WordCharLSTM(nn.Module):
    def __init__(
            self,
            num_word_embeddings,
            num_tags,
            word_embeddings,
            num_char_embeddings,
            word_lstm,
            char_lstm,
            char_padding_idx=0,
            train_word_embeddings=False):
        super(WordCharLSTM, self).__init__()

        self.char_embeddings = nn.Embedding(
            num_embeddings=num_char_embeddings,
            embedding_dim=CHAR_EMBEDDING_DIM,
            padding_idx=char_padding_idx)

        self.word_embeddings = nn.Embedding(
            num_embeddings=num_word_embeddings,
            embedding_dim=WORD_EMBEDDING_DIM,
            padding_idx=PAD_IDX,
            _weight=word_embeddings)

        if word_embeddings:
            self.word_embeddings.weight.requires_grad = train_word_embeddings

        self.char_lstm = char_lstm
        self.embedding_dropout = nn.Dropout(0.3)
        self.word_lstm = word_lstm
        self.output_dropout = nn.Dropout(0.3)
        self.out = nn.Linear(WORD_LSTM_HIDDEN_SIZE, num_tags)

        nn.init.xavier_uniform_(self.out.weight)
        for name, param in self.word_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        for name, param in self.char_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    # TODO : maybe other initialization methods?
    def init_hidden(self, batch_size):  # initialize hidden states
        h = zeros(WORD_LSTM_NUM_LAYERS * WORD_LSTM_NUM_DIRS,
                  batch_size,
                  WORD_LSTM_HIDDEN_SIZE // WORD_LSTM_NUM_DIRS)  # hidden states
        c = zeros(WORD_LSTM_NUM_LAYERS * WORD_LSTM_NUM_DIRS,
                  batch_size,
                  WORD_LSTM_HIDDEN_SIZE // WORD_LSTM_NUM_DIRS)  # cell states
        return (h, c)

    def forward(self, word_x, mask, char_x):
        char_output = self._char_forward(char_x)
        batch_size = word_x.size(0)
        max_seq_len = word_x.size(1)
        char_output = char_output.reshape(batch_size, max_seq_len, -1)  # last dimension is for char lstm hidden size

        word_x = self.word_embeddings(word_x)
        word_x = torch.cat([word_x, char_output], -1)
        word_x = self.embedding_dropout(word_x)

        initial_hidden = self.init_hidden(batch_size)  # batch size is first
        word_x = nn.utils.rnn.pack_padded_sequence(word_x, mask.sum(1).int(), batch_first=True)
        output, hidden = self.word_lstm(word_x, initial_hidden)

        output, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.output_dropout(output)
        output = self.out(output)  # batch x seq_len x num_tags
        output *= mask.unsqueeze(-1)  # mask - batch x seq_len -> batch x seq_len x 1
        return output

    def _char_forward(self, x):
        word_lengths = x.gt(0).sum(1)  # actual word lengths
        sorted_padded, order = _sort(x, word_lengths)
        embedded = self.char_embeddings(sorted_padded)

        word_lengths_copy = word_lengths.clone()
        word_lengths_copy[word_lengths == 0] = 1
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, word_lengths_copy[order], True)
        packed_output, _ = self.char_lstm(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, True)

        _, reverse_sort_order = torch.sort(order, dim=0)
        output = output[reverse_sort_order]

        indices_of_lasts = (word_lengths_copy - 1).unsqueeze(1).expand(-1, output.shape[2]).unsqueeze(1)
        output = output.gather(1, indices_of_lasts).squeeze()
        output[word_lengths == 0] = 0
        return output


class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.transition = nn.Parameter(randn(num_tags, num_tags))
        self.transition.data[START_TAG_IDX, :] = -10000.  # no transition to START
        self.transition.data[:, STOP_TAG_IDX] = -10000.  # no transition from END except to PAD
        self.transition.data[:, PAD_IDX] = -10000.  # no transition from PAD except to PAD
        self.transition.data[PAD_IDX, :] = -10000.  # no transition to PAD except from END
        self.transition.data[PAD_IDX, STOP_TAG_IDX] = 0.
        self.transition.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, h, mask):
        # initialize forward variables in log space
        alpha = Tensor(h.shape[0], self.num_tags).fill_(-10000.)  # [B, S]
        # TODO: pytorch tutorial says wrap it in a variable to get automatic backprop, do we need it here? to be checked
        alpha[:, START_TAG_IDX] = 0.

        transition = self.transition.unsqueeze(0)  # [1, S, S]
        for t in range(h.size(1)):  # iterate through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emission = h[:, t].unsqueeze(2)  # [B, S, 1]
            alpha_t = log_sum_exp(alpha.unsqueeze(1) + emission + transition)  # [B, 1, S] -> [B, S, S] -> [B, S]
            alpha = alpha_t * mask_t + alpha * (1 - mask_t)

        Z = log_sum_exp(alpha + self.transition[STOP_TAG_IDX])
        return Z  # partition function

    def score(self, h, y, mask):  # calculate the score of a given sequence
        batch_size = h.shape[0]
        score = Tensor(batch_size).fill_(0.)
        # TODO: maybe instead of unsqueezing following two separately do it after sum in line for score calculation
        # TODO: check if unsqueezing needed at all
        h = h.unsqueeze(3)
        transition = self.transition.unsqueeze(2)
        y = torch.cat([LongTensor([START_TAG_IDX]).view(1, -1).expand(batch_size, 1), y], 1)  # add start tag to begin
        # TODO: the loop can be vectorized, probably
        for t in range(h.size(1)):  # iterate through the sequence
            mask_t = mask[:, t]
            emission = torch.cat([h[i, t, y[i, t + 1]] for i in range(batch_size)])
            transition_t = torch.cat([transition[seq[t + 1], seq[t]] for seq in y])
            score += (emission + transition_t) * mask_t
        # get transitions from last tags to stop tag: use gather to get last time step
        lengths = mask.sum(1).long()
        indices = lengths.unsqueeze(1)  # we can safely use lengths as indices, because we prepended start tag to y
        last_tags = y.gather(1, indices).squeeze()
        score += self.transition[STOP_TAG_IDX, last_tags]
        return score

    def decode(self, h, mask):  # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        backpointers = LongTensor()
        batch_size = h.shape[0]
        delta = Tensor(batch_size, self.num_tags).fill_(-10000.)
        delta[:, START_TAG_IDX] = 0.

        # TODO: is adding stop tag within loop needed at all???
        # pro argument: yes, backpointers needed at every step - to be checked
        for t in range(h.size(1)):  # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            mask_t = mask[:, t].unsqueeze(1)
            # TODO: maybe unsqueeze transition explicitly for 0 dim for clarity
            next_tag_var = delta.unsqueeze(1) + self.transition  # B x 1 x S + S x S
            delta_t, backpointers_t = next_tag_var.max(2)
            backpointers = torch.cat((backpointers, backpointers_t.unsqueeze(1)), 1)
            delta_next = delta_t + h[:, t]  # plus emission scores
            delta = mask_t * delta_next + (1 - mask_t) * delta  # TODO: check correctness
            # for those that end here add score for transitioning to stop tag
            if t + 1 < h.size(1):
                # mask_next = mask[:, t + 1].unsqueeze(1)
                # ending = mask_next.eq(0.).float().expand(batch_size, self.num_tags)
                # delta += ending * self.transition[STOP_TAG_IDX].unsqueeze(0)
                # or
                ending_here = (mask[:, t].eq(1.) * mask[:, t + 1].eq(0.)).view(1, -1).float()
                delta += ending_here.transpose(0, 1).mul(self.transition[STOP_TAG_IDX])  # add outer product of two vecs
                # TODO: check equality of these two again

        # TODO: should we add transition values for getting in stop state only for those that end here?
        # TODO: or to all?
        delta += mask[:, -1].view(1, -1).float().transpose(0, 1).mul(self.transition[STOP_TAG_IDX])
        best_score, best_tag = torch.max(delta, 1)

        # back-tracking
        backpointers = backpointers.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for idx in range(batch_size):
            prev_best_tag = best_tag[idx]  # best tag id for single instance
            length = int(scalar(mask[idx].sum()))  # length of instance
            for backpointers_t in reversed(backpointers[idx][:length]):
                prev_best_tag = backpointers_t[prev_best_tag]
                best_path[idx].append(prev_best_tag)
            best_path[idx].pop()  # remove start tag
            best_path[idx].reverse()

        return best_path


def create_crf_on_lstm_model(
        word_vocab_size,
        tag_vocab_size,
        char_vocab_size,
        word_embeddings):
    char_lstm = nn.LSTM(
        input_size=CHAR_EMBEDDING_DIM,
        hidden_size=CHAR_LSTM_HIDDEN_SIZE // CHAR_LSTM_NUM_DIRS,
        num_layers=CHAR_LSTM_NUM_LAYERS,
        bias=True,
        batch_first=True,
        bidirectional=CHAR_LSTM_BIDIRECTIONAL)

    word_lstm = nn.LSTM(
        input_size=WORD_EMBEDDING_DIM + CHAR_LSTM_HIDDEN_SIZE,
        hidden_size=WORD_LSTM_HIDDEN_SIZE // WORD_LSTM_NUM_DIRS,
        num_layers=WORD_LSTM_NUM_LAYERS,
        bias=True,
        batch_first=True,
        bidirectional=WORD_LSTM_BIDIRECTIONAL)

    return CRFOnLSTM(
        word_vocab_size,
        tag_vocab_size,
        torch.tensor(word_embeddings, dtype=torch.float) if word_embeddings else None,
        char_vocab_size,
        word_lstm,
        char_lstm)


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
