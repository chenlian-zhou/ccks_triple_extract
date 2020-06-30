# encoding: utf-8


from collections import defaultdict

from data_utils import load_vocab_vectors, UNK, _get_spans
from model import create_crf_on_lstm_model
from train import _load_words_tags_chars, _load_data, batch_generator
from utils import *


def _predict():
    word2idx, tag2idx, char2idx = _load_words_tags_chars("data/words.txt", "data/tags.txt", "data/chars.txt")

    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    # embeddings = load_vocab_vectors("data/filtered_embeddings.txt")

    model = create_crf_on_lstm_model(len(word2idx), len(tag2idx), len(char2idx) + 1, None)

    model.eval()
    print(model)
    load_checkpoint('model.epoch3-0.7339571436404638', model)

    arrays = _load_data('data/eng.testb', word2idx, tag2idx, char2idx)

    seen_entities = set()

    type2referenced = defaultdict(set)
    type2predicted = defaultdict(set)
    mistakes = 0
    sentenceIdx = 0
    for word_x, char_x, y in batch_generator(*arrays):
        y_pred = model.decode(word_x, char_x)

        for xi, yi, yp in zip(word_x, y, y_pred):
            text, predicted_tags = _instance_from(yp, xi, idx2word, idx2tag)
            _, original_tags = _instance_from(yi, xi, idx2word, idx2tag)
            length = int(xi.gt(0).sum())  # TODO: replace check with 0 with check for pad index
            referenced = _get_spans(text[:length], original_tags[:length])
            try:
                predicted = _get_spans(text[:length], predicted_tags[:length])

            except ValueError:
                mistakes += 1
                continue

            for reference in referenced:
                type2referenced[reference[2]].add((sentenceIdx, *reference))

            for prediction in predicted:
                type2predicted[prediction[2]].add((sentenceIdx, *prediction))
                seen_entities.add(prediction[3])

            sentenceIdx += 1

    print('recalls')
    for type, referenced in type2referenced.items():
        print(type, len(referenced), len(referenced.intersection(type2predicted[type])),
              float(len(referenced.intersection(type2predicted[type]))) / len(referenced))

    print()
    print('precisions')
    for type, predicted in type2predicted.items():
        print(type, len(predicted), len(predicted.intersection(type2referenced[type])),
              float(len(predicted.intersection(type2referenced[type]))) / len(predicted))

    print(mistakes, 'mistakes')


def _instance_from(predicted, entry, idx2word, idx2tag):
    words = [idx2word.get(idx.item(), UNK) for idx in entry]
    tags = [idx2tag.get(idx.item()) if isinstance(idx, torch.Tensor) else idx2tag[idx] for idx in predicted]
    return words, tags


if __name__ == "__main__":
    print("cuda: {}".format(CUDA))
    _predict()
