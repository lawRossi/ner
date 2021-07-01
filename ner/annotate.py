import ahocorasick
from itertools import groupby
from operator import itemgetter


def load(path):
    with open(path, encoding="utf-8") as fi:
        return [line.strip() for line in fi]


def make_automation(lookup_paths):
    """制定不同类型的实体文件，读取实体并构建AC自动机

    Args:
        lookup_paths (dict): 实体类型与实体文件的映射

    Returns:
        ahocorasick.Automaton: 构建的AC自动机
    """
    auto_maton = ahocorasick.Automaton()
    for type_, path in lookup_paths.items():
        for term in load(path):
            auto_maton.add_word(term, (term, type_))
    auto_maton.make_automaton()
    return auto_maton


def longest_match(matches):
    """获取最长匹配

    Args:
        matches (list): 匹配列表

    Yields:
        tupple: 最长匹配
    """
    matches = [(match[0]-len(match[1][0])+1, match[1]) for match in matches]
    matches = sorted(matches, key=itemgetter(0))
    for _, match_set in groupby(matches, itemgetter(0)):
        yield max(match_set, key=lambda x: len(x[1][0]))


def annotate(auto_maton, text):
    """自动标注文本的实体。
    标注时使用AC自动机进行匹配，同一位置有多个匹配时使用最长匹配，标注格式为markdown格式。

    Args:
        A (ahocorasick.Automaton): AC自动机
        text (str): 待标注的文本

    Returns:
        str: 标注后的文本
    """
    current = -1
    annotated_text = []
    for pos, match in longest_match(auto_maton.iter(text)):
        if pos <= current:
            continue
        annotated_text.extend(text[current+1:pos])
        annotation = f"[{match[0]}]({match[1]})"
        annotated_text.extend(annotation)
        current = pos + len(match[0]) - 1
    annotated_text.extend(text[current+1:])
    return "".join(annotated_text)


def annotate_file(source_path, save_path, lookup_paths):
    """标注文本文件

    Args:
        source_path (str): 待标注文本文件路径
        save_path (str): 保存标注结果的文件路径
        lookup_paths (dict): 实体类别与实体文件的映射
    
    Examples:

        lookup_paths = {
            "person": "data/persons.txt",
            "country": "data/countries.txt",
        }
        annotate_file("data/text.txt", "text.md", lookup_paths)
    """
    auto_maton = make_automation(lookup_paths)
    with open(source_path, encoding="utf-8") as fi, \
            open(save_path, "w", encoding="utf-8") as fo:
        for line in fi:
            new_line = annotate(auto_maton, line.strip())
            fo.write(new_line)
            fo.write("\n")
