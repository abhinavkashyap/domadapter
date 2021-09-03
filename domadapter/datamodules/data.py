from typing import List


class InputExample(object):
    """
    Taken from https://github.com/srush/transformers/blob/master/examples/utils_ner.py
    A single training/test example for token classification.
    """

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid: str = guid
        self.words: List[str] = words
        self.labels: List[str] = labels


class InputFeatures(object):
    """
    Taken from https://github.com/srush/transformers/blob/master/examples/utils_ner.py
    A single set of features of data.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
