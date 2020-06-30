# encoding: utf8

import os

import numpy
import numpy as np
import pandas
from sklearn.utils import shuffle

from utils import LongTensor

PAD = "$PAD$"
UNK = "$UNK$"
NUM = "NUM"
NONE = "O"
STOP_TAG = "<EOS>"  # end of sequence
START_TAG = "<SOS>"  # start of sequence
# UNK = "<UNK>"  # unknown token

PAD_IDX = 0
START_TAG_IDX = 1
STOP_TAG_IDX = 2
# UNK_IDX = 3

TOKEN2IDX = {
    PAD: PAD_IDX,
    START_TAG: START_TAG_IDX,
    STOP_TAG: STOP_TAG_IDX
}


class CoNLLDataset(object):
    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None):
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename, encoding='utf8') as f:
            words, tags = (), ()
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = (), ()
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += (word,)
                    tags += (tag,)

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    print("Building vocabulary")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(words):
    vocab_char = set()
    for word in words:
        vocab_char.update(word)

    return vocab_char


def get_embedding_vocab(filename):
    print("Loading embeddings vocabulary from file")
    vocab = set()
    with open(filename, encoding='utf8') as f:
        for line in f:
            vocab.add(line.strip().split(' ')[0])
    print("done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    print("Writing vocabulary to file")
    with open(filename, "w", encoding='utf8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("done. {} tokens".format(len(vocab)))


def load_vocab(filename, start_from=0):
    result = dict()
    with open(filename, encoding='utf8') as f:
        for idx, word in enumerate(f, start_from):
            result[word.strip()] = idx

    return result


def filter_embeddings_in_vocabulary(vocab_filename, embeddings_filename, filtered_embeddings_filename):
    vocab = load_vocab(vocab_filename)
    with open(embeddings_filename, encoding='utf8') as f:
        __, dim = next(f).strip().split()
        embeddings = np.zeros([len(vocab), int(dim)])
        print(embeddings.shape)
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.save(filtered_embeddings_filename, embeddings)


def load_vocab_vectors(filename):
    return np.load(filename + '.npy')  # wtf?


def get_processing_word(vocab_words=None,
                        vocab_chars=None,
                        lowercase=False,
                        chars=False,
                        allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """

    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = ()
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += (vocab_chars[char],)

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def get_processing_tag(tag2idx):
    def f(tag):
        if tag in tag2idx:
            return tag2idx[tag]
        else:
            raise ValueError("{} was not found in tag list".format(tag))

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length


def _load_words_tags_chars(words_file, tags_file, chars_file):
    word2idx = load_vocab(words_file)
    char2idx = load_vocab(chars_file, 1)

    tag2idx = load_vocab(tags_file, 3)  # since we have 3 more tokens, see below
    tag2idx.update(TOKEN2IDX)
    # tag2idx[PAD] = TOKEN2IDX[PAD]
    # tag2idx[START_TAG] = TOKEN2IDX[START_TAG]
    # tag2idx[STOP_TAG] = TOKEN2IDX[STOP_TAG]
    return word2idx, tag2idx, char2idx


def _load_data(training_filename, word2idx, tag2idx, char2idx):
    # TODO: right now it just relies on the fact that there is UNK in vocab, has to make sure it exists
    processing_word = get_processing_word(word2idx,
                                          char2idx,
                                          lowercase=False,
                                          chars=True)
    processing_tag = get_processing_tag(tag2idx)

    word_indices_and_tag_indices = CoNLLDataset(training_filename, processing_word, processing_tag)
    # TODO: maybe use explicit transformation for clarity
    word_id_lists, tag_id_lists = list(zip(*word_indices_and_tag_indices))
    char_id_lists, word_id_lists = list(zip(*[list(zip(*couple)) for couple in word_id_lists]))
    seq_lengths_in_words = numpy.array([len(i) for i in word_id_lists])
    word_lengths = numpy.array([numpy.array([len(word_char_seq) for word_char_seq in sequence])
                                for sequence in char_id_lists])
    return word_id_lists, tag_id_lists, char_id_lists, seq_lengths_in_words, word_lengths


def batch_generator(*arrays, batch_size=32):
    word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths = arrays
    word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths = \
        shuffle(word_id_lists, tag_id_lists, char_id_lists, seq_length_in_words, word_lengths)

    num_instances = len(word_id_lists)
    batch_count = int(numpy.ceil(num_instances / batch_size))
    from tqdm import tqdm
    prog = tqdm(total=num_instances)
    for idx in range(batch_count):
        startIdx = idx * batch_size
        endIdx = (idx + 1) * batch_size if (idx + 1) * batch_size < num_instances else num_instances
        batch_lengths = seq_length_in_words[startIdx:endIdx]
        batch_maxlen = batch_lengths.max()
        argsort = numpy.argsort(batch_lengths)[::-1].copy()  # without the copy torch complains about negative strides
        char_batch = numpy.array(char_id_lists[startIdx:endIdx])[argsort]

        # make each sentence in batch contain same number of words
        char_batch = [sentence + ((0,),) * (batch_maxlen - len(sentence)) for sentence in char_batch]

        word_lengths = [len(word) for sentence in char_batch for word in sentence]
        max_word_length = max(word_lengths)
        # make each word in batch contain same number of chars
        chars = [word + (0,) * (max_word_length - len(word)) for sentence in char_batch for word in sentence]
        chars = LongTensor(chars)

        words = LongTensor([word_ids + (PAD_IDX,) * (batch_maxlen - len(word_ids))
                            for word_ids in word_id_lists[startIdx:endIdx]])
        tags = LongTensor([tag_ids + (PAD_IDX,) * (batch_maxlen - len(tag_ids))
                           for tag_ids in tag_id_lists[startIdx:endIdx]])

        prog.update(len(batch_lengths))
        yield words[argsort], chars, tags[argsort]
    prog.close()


def get_entities_and_types_per_text(dataset):
    texts = []
    types = []
    entities = []

    for words, tags in dataset:
        _types = set()
        _entities = []
        _entity = []
        startIdx = None
        # TODO: use_get_spans function
        for i, (word, tag) in enumerate(zip(words, tags)):
            if tag != 'O':
                if tag.startswith('B'):
                    if _entity:
                        _entities.append((startIdx, i, _type, " ".join(_entity)))
                    _entity = [word]
                    _type = tag[2:]
                    _types.add(_type)
                    startIdx = i
                else:
                    _entity.append(word)

            else:
                if _entity:
                    _entities.append((startIdx, i, _type, " ".join(_entity)))
                    _entity = []

        if _entity:
            _entities.append((startIdx, i + 1, _type, " ".join(_entity)))

        _entities = tuple(_entities)
        entities.append(_entities)
        types.append(tuple(sorted(_types)))
        texts.append(" ".join(words))

    return entities, types, texts


def _get_spans(words, tags):
    _types = set()
    _entities = []
    _entity = []
    startIdx = None

    for i, (word, tag) in enumerate(zip(words, tags)):
        if tag != 'O':
            if tag.startswith('B'):
                if _entity:
                    _entities.append((startIdx, i, _type, " ".join(_entity)))
                _entity = [word]
                _type = tag[2:]
                _types.add(_type)
                startIdx = i
            else:
                if _entity:
                    if _type == tag[2:]:
                        _entity.append(word)
                    else:
                        _entities.append((startIdx, i, _type, " ".join(_entity)))
                        _entity = [word]
                        _type = tag[2:]
                        _types.add(_type)
                        startIdx = i
                else:
                    _entity = [word]
                    _type = tag[2:]
                    _types.add(_type)
                    startIdx = i
                # if not _entity:
                #     raise ValueError("Illegal state, current tag - {}".format(tag))
                # _entity.append(word)
        else:
            if _entity:
                _entities.append((startIdx, i, _type, " ".join(_entity)))
                _entity = []

    if _entity:
        _entities.append((startIdx, i + 1, _type, " ".join(_entity)))
    return _entities


def get_df():
    entities, types, texts = get_entities_and_types_per_text(
        CoNLLDataset(os.path.join(os.getcwd(), 'data/more_annotated.txt'), get_processing_word()))
    return pandas.DataFrame(data={'texts': texts, 'entities': entities, 'types': types})


if __name__ == '__main__':
    entities, types, texts = get_entities_and_types_per_text(
        CoNLLDataset('data/more_annotated_train.conll', get_processing_word()))
    train_entities = set()
    for _entities in entities:
        if _entities:
            for entity in _entities:
                train_entities.add(entity[3])

    entities, types, texts = get_entities_and_types_per_text(
        CoNLLDataset('data/more_annotated_test.conll', get_processing_word()))
    test_entities = set()
    for _entities in entities:
        if _entities:
            for entity in _entities:
                test_entities.add(entity[3])

    print(len(test_entities))
    print(len(train_entities))
    print(len(test_entities.difference(train_entities)))
    print(test_entities.difference(train_entities))
