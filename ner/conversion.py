# -*- coding:utf-8 -*-
import re


def brat2conll(src_file_frefix, save_file):
    """Brat标注方式转化为Conll标注方式

    Args:
        src_file_frefix (str): Brat标注方式有两个文件，这个参数指定文件的前缀
        save_file (str): 保留转化结果的文件
    """
    with open(src_file_frefix+'.txt', encoding='utf-8') as fi:
        text = fi.read().strip()
    annotation = ['O'] * len(text)
    with open(src_file_frefix+'.ann', encoding='utf-8') as fi:
        prev_start = -1
        prev_end = -1
        for line in fi:
            splits = line.strip().split('\t')
            [_type, start, end] = splits[1].split(" ")
            start = int(start)
            end = int(end)
            if prev_start <= start and end <= prev_end:  # overlap
                continue
            annotation[start] = 'B_%s' % _type
            for i in range(start+1, end-1):
                annotation[i] = 'I_%s' % _type
            annotation[end-1] = 'E_%s' % _type
            prev_start = start
            prev_end = end
    with open(save_file, 'w', encoding='utf-8') as fo:
        assert len(text) == len(annotation)
        for chr, tag in zip(text, annotation):
            if chr == '\n':
                fo.write('\n')
                continue
            else:
                fo.write('%s\t%s\n' % (chr, tag))


def markdown2conll(source_path, save_path):
    p = re.compile("\[(?P<word>.+?)\]\((?P<type>.+?)\)")
    with open(source_path, encoding="utf-8") as fi, \
            open(save_path, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            start = 0
            for match in p.finditer(line):
                for i in range(start, match.start()):
                    fo.write(f"{line[i]}\tO\n")
                word = match.group("word")
                type_ = match.group("type")
                if len(word) == 1:
                    fo.write(f"{word}\tS_{type_}\n")
                else:
                    fo.write(f"{word[0]}\tB_{type_}\n")
                    for chr in word[1:-1]:
                        fo.write(f"{chr}\tI_{type_}\n")
                    fo.write(f"{word[-1]}\tE_{type_}\n")
                start = match.end()
            for i in range(start, len(line)):
                fo.write(f"{line[i]}\tO\n")
            fo.write("\n")


def conll2brat(source_path, save_path_prefix):
    with open(source_path, encoding="utf-8") as fi:
        text = []
        anns = []
        entity = None
        type_ = None
        entity_index = 1
        index = -1
        start = -1
        for line in fi:
            index += 1
            line = line.strip()
            if not line:
                text.append('\n')
            else:
                splits = line.split('\t')
                text.append(splits[0])
                if splits[1].startswith("B"):
                    start = index
                    entity = [splits[0]]
                    type_ = splits[1][2:]
                elif splits[1].startswith("I"):
                    entity.append(splits[0])
                elif splits[1].startswith("E"):
                    entity.append(splits[0])
                    anns.append(f'T{entity_index}\t{type_} {start} {index+1}\t{"".join(entity)}')
                    entity_index += 1
                    entity = []
                elif splits[1].startswith("S"):
                    type_ = splits[1][2:]
                    anns.append(f'T{entity_index}\t{type_} {index} {index+1}\t{splits[0]}')
                    entity_index += 1
                    entity = []
        with open(f"{save_path_prefix}.txt", "w", encoding="utf-8") as fo:
            fo.write("".join(text))
    
        with open(f"{save_path_prefix}.ann", "w", encoding="utf-8") as fo:
            for ann in anns:
                fo.write(ann)
                fo.write("\n")


def conll2markdown(source_path, save_path):
    with open(source_path, encoding="utf-8") as fi, \
            open(save_path, "w", encoding="utf-8") as fo:
        sentence = []
        entity = None
        type_ = None
        start = -1
        for line in fi:
            line = line.strip()
            if not line:
                fo.write("".join(sentence))
                fo.write("\n")
                sentence = []
            else:
                splits = line.split('\t')
                if splits[1] == "O":
                    sentence.append(splits[0])
                elif splits[1].startswith("B"):
                    entity = [splits[0]]
                    type_ = splits[1][2:]
                elif splits[1].startswith("I"):
                    entity.append(splits[0])
                elif splits[1].startswith("E"):
                    entity.append(splits[0])
                    sentence.extend(f"[{''.join(entity)}]({type_})")
                    entity = []
                elif splits[1].startswith("S"):
                    type_ = splits[1][2:]
                    sentence.extend(f"[{splits[0]}]({type_})")
                    entity = []
