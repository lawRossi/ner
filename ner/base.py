class NerTagger():
    """命名实体识别包装类的基类，用于定义标准接口
    """

    @classmethod
    def load_model(cls, model_dir):
        raise NotImplementedError
    
    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts):
        raise NotImplementedError

    def recognize_nes(self, text):
        """识别实体

        Args:
            text (str): 输入文本

        Returns:
            list: 识别到的实体列表
        """
        tags = self.predict(text)
        nes = []
        ne = []
        start = None
        type_ = None
        for i, (chr, tag) in enumerate(zip(text, tags)):
            if tag == 'O':
                ne, type_ = self._handle_current_entity(nes, ne, start, i, type_)
            elif tag.startswith('S'):
                ne, type_ = self._handle_current_entity(nes, ne, start, i, type_)
                start = i
                type_ = tag[2:]
                ne.append(chr)
            elif tag.startswith('B'):
                ne, type_ = self._handle_current_entity(nes, ne, start, i, type_)
                start = i
                type_ = tag[2:]
                ne.append(chr)
            elif tag.startswith('E'):
                ne.append(chr)
                ne, type_ = self._handle_current_entity(nes, ne, start, i+1, type_)
            else:
                ne.append(chr)
        if ne and type_ is not None:
            nes.append(("".join(ne), type_, (start, len(text))))
        return nes

    def _handle_current_entity(self, nes, ne, start, end, type_):
        if ne and type_ is not None:
            nes.append(("".join(ne), type_, (start, end)))
        ne = []
        type_ = None
        return ne, type_
