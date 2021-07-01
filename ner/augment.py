import random
import re


def load(path):
    with open(path, encoding="utf-8") as fi:
        return [line.strip() for line in fi]


def augment(sample, lookup_tables, aug_num):
    """数据扩充。
    找到文本里标注的每个实体，随机替换成同类型的其他实体。

    Args:
        sample (str): 带标注的文本样本
        lookup_tables (dict): 实体类型与实体列表的映射
        aug_num (int): 每个实体扩充的个数

    Returns:
        list: 扩充后的样本
    """
    p = re.compile("\[(?P<word>.+?)\]\((?P<type>.+?)\)")
    new_samples = []
    for _ in range(aug_num):
        replacements = []
        for match in p.finditer(sample):
            type_ = match.group("type")
            word = match.group("word")
            choice = random.choice(lookup_tables[type_])
            if choice != word:
                replacements.append((match, choice, type_))
        current_index = 0
        new_sample = []
        for replacement in replacements:
            match, word, type_ = replacement
            while current_index < match.start():
                new_sample.append(sample[current_index])
                current_index += 1
            repstr = f"[{word}]({type_})"
            new_sample.extend(repstr)
            current_index += (match.end() - match.start())
        new_sample.extend(sample[current_index:])
        new_samples.append("".join(new_sample))
    return new_samples


def generate_augment_data(source_path, save_path, lookup_paths, aug_num=3):
    """生成扩充数据

    Args:
        source_path (str)): 待扩充的标注文本文件
        save_path (str): 保存结果的文件
        lookup_paths (dict): 实体类型与实体文件的映射
        aug_num (int, optional): 每个实体扩充的个数
    
    Examples:

        lookup_paths = {
            "person": "data/persons.txt",
            "country": "data/countries.txt",
        }
        generate_augment_data("data/text.md", "data/aug_text.md", lookup_paths, 5)
    """
    lookup_tables = {}
    for key, path in lookup_paths.items():
        lookup_tables[key] = load(path)
    samples = load(source_path)
    with open(save_path, "w", encoding="utf-8") as fo:
        for sample in samples:
            fo.write(sample)
            fo.write("\n")
            for new_sample in augment(sample, lookup_tables, aug_num):
                fo.write(new_sample)
                fo.write("\n")
