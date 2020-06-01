class IText:
    def split_to_sentences(self):
        raise NotImplementedError

    def tokenize(self):
        raise NotImplementedError

    def filter_stop_words(self, stop_words=None):
        raise NotImplementedError