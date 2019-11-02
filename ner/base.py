import os
import time


class NerTagger():
    def __init__(self, data_processor, converter):
        self.data_processor = data_processor
        self.converter = converter
    
    def train(self, args):
        raise NotImplementedError

    def test(self, args):
        raise NotImplementedError

    @classmethod
    def load_model(cls, model_dir):
        raise NotImplementedError
    
    def save_model(self, model_dir, args=None):
        raise NotImplementedError
    
    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts):
        raise NotImplementedError

    def recognize_nes(self, text):
        tags = self.predict(text)
        nes = []
        ne = []
        start = None
        type_ = None
        for i, (chr, tag) in enumerate(zip(text, tags)):
            if tag == 'O':
                if ne:
                    nes.append(("".join(ne), type_, (start, i)))
                    ne = []
            elif tag.startswith('S'):
                if ne:
                    nes.append(("".join(ne), type_, (start, i)))
                start = i
                type_ = tag[2:]
                ne = [chr]
            elif tag.startswith('B'):
                if ne:
                    nes.append(("".join(ne), type_, (start, i)))
                start = i
                type_ = tag[2:]
                ne = [chr]
            elif tag.startswith('E'):
                ne.append(chr)
                nes.append(("".join(ne), type_, (start, i+1)))
                ne = []
            else:
                ne.append(chr)
        if ne:
            nes.append(("".join(ne), type_, (start, len(text))))
        return nes
    
    def save_checkpoint(self, model_dir, args=None):
        checkpoint = f"checkpoint_{int(time.time())}"
        checkpoint_dir = os.path.join(model_dir, checkpoint)
        os.makedirs(checkpoint_dir)
        self.save_model(checkpoint_dir, args)
