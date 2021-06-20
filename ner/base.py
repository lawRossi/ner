class NerTagger():

    @classmethod
    def load_model(cls, model_dir):
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
                ne, type_ = self.handle_current_entity(nes, ne, start, i, type_)
            elif tag.startswith('S'):
                ne, type_ = self.handle_current_entity(nes, ne, start, i, type_)
                start = i
                type_ = tag[2:]
                ne.append(chr)
            elif tag.startswith('B'):
                ne, type_ = self.handle_current_entity(nes, ne, start, i, type_)
                start = i
                type_ = tag[2:]
                ne.append(chr)
            elif tag.startswith('E'):
                ne.append(chr)
                ne, type_ = self.handle_current_entity(nes, ne, start, i+1, type_)
            else:
                ne.append(chr)
        if ne and type_ is not None:
            nes.append(("".join(ne), type_, (start, len(text))))
        return nes

    def handle_current_entity(self, nes, ne, start, end, type_):
        if ne and type_ is not None:
            nes.append(("".join(ne), type_, (start, end)))
        ne = []
        type_ = None
        return ne, type_
