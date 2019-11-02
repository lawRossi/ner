# -×- coding:utf-8 -*-
from ..base import NerTagger
from itertools import chain
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from collections import Counter
import codecs
from ..ner_evaluate import evaluate


class CrfNerTagger(NerTagger):
    """
    NER tagger based on crf
    """
    BOS = "BOS"
    EOS = "EOS"

    def __init__(self, data_processor=None, converter=None):
        super().__init__(data_processor, converter)
        self.tagger = None

    def load(self, path):
        with codecs.open(path, encoding="utf-8") as fi:
            return [line.strip() for line in fi]

    def load_training_samples(self, path):
        samples = []
        sample = []
        with codecs.open(path, encoding="utf-8") as fi:
            for line in fi:
                line = line.strip()
                if not line:
                    samples.append(sample)
                    sample = []
                    continue

                splits = line.split("\t")
                sample.append((splits[0], splits[-1]))

        print("finished loading %d smaples" % len(samples))
        return samples

    def is_ch_number(self, c):
        return c in [u"零", u"一", u"二", u"三", u"四", u"五", u"六", u"七", u"八",
                     u"九", u"十", u"百", u"千", u"万"]

    def is_punc(self, c):
        return c in [u"，", u"、", u",", u"/"]

    def token2features(self, sample, i):
        character = sample[i][0]
        features = [
            'character=' + character,
            'character.isdigit=%s' % character.isdigit(),
            'character.is_ch_number=%s' % self.is_ch_number(character),
            'character.is_alpha=%s' % character.isalpha(),
            'character.is_punc=%s' % self.is_punc(character)
        ]
        if i > 0:
            character = sample[i-1][0]
            features.extend([
                '-1character=' + character,
                '-1:0character=%s%s' %(sample[i-1][0], sample[i][0]),
                '-1character.isdigit=%s' % character.isdigit(),
                '-1character.is_ch_number=%s' % self.is_ch_number(character),
                '-1character.is_alpha=%s' % character.isalpha(),
            ])
        
        else:
            features.append(self.BOS)

        if i > 1:
            features.extend([
                '-2character=' + sample[i-1][0],
                '-2:-1character=%s%s' % (sample[i-2][0], sample[i-1][0]),
            ])

        if i < len(sample) - 1:
            character = sample[i+1][0]
            features.extend([
                '+1character=' + character,
                '+1character.isdigit=%s' % character.isdigit(),
                # '+1character.is_ch_number=%s' % self.is_ch_number(character),
                '+1character.is_alpha=%s' % character.isalpha()
            ])
        else:
            features.append(self.EOS)

        if i < len(sample) - 2:
            features.extend([
                '+2character=' + sample[i+1][0],
                '+1:+2character=%s%s' % (sample[i+1][0], sample[i+2][0]),
            ])

        return features

    def sample2features(self, sample):
        return [self.token2features(sample, i) for i in range(len(sample))]

    def sample2labels(self, sample):
        return [label for token, label in sample]

    def sample2tokens(self, sample):
        return [token for token, label in sample]

    def sentence2features(self, sentence):
        sample = [(c, "") for c in sentence]
        return self.sample2features(sample)

    def train(self, sample_path, model_path='ner.crfsuite'):
        samples = self.load_training_samples(sample_path)
        self._train(samples, model_path)

    def _train(self, samples, model_path):
        X_train = [self.sample2features(s) for s in samples]
        y_train = [self.sample2labels(s) for s in samples]
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 500,  # stop earlier
            # include transitions that are possible, but not observed
            # 'feature.possible_transitions': True
        })
        trainer.train(model_path)
        print(trainer.logparser.last_iteration)
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_path)

    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.
        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )

    def test(self, sample_path):
        test_samples = self.load_training_samples(sample_path)
        self._test(test_samples)

    def _test(self, samples):
        X_test = [self.sample2features(s) for s in samples]
        y_test = [self.sample2labels(s) for s in samples]
        y_pred = [self.tagger.tag(xseq) for xseq in X_test]
        # print(self.bio_classification_report(y_test, y_pred))
        # self.print_transitions(self.tagger)
        evaluate(y_test, y_pred)

    def predict(self, text):
        seq = self.sentence2features(text)
        return self.tagger.tag(seq)

    def print_transitions(self, tagger):
        info = tagger.info()
        def do_print(trans_features):
            for (label_from, label_to), weight in trans_features:
                print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        print("Top likely transitions:")
        do_print(Counter(info.transitions).most_common(15))
        print("\nTop unlikely transitions:")
        do_print(Counter(info.transitions).most_common()[-15:])

    def print_state_features(self, tagger):
        info = tagger.info()
        def do_print(state_features ):
            for (attr, label), weight in state_features:
                print("%0.6f %-6s %s" % (weight, label, attr))
        print("Top positive:")
        do_print(Counter(info.state_features).most_common(20))
        print("\nTop negative:")
        do_print(Counter(info.state_features).most_common()[-20:])

    def load_model(self, model_path='address.crfsuite'):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_path)


if __name__ == '__main__':
    # tagger = NerTagger()
    # tagger.train('../data/test.conll', "../data/ner.crfsuite")
    tagger = CrfNerTagger()
    tagger.load_model("data/ner.crfsuite")
    print(tagger.recognize_nes("梅西进球了,内马尔助攻"))
    print(tagger.recognize_nes("罗也进球了，风终于打败了马"))
