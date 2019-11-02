import random
import re


def load(path):
    with open(path, encoding="utf-8") as fi:
        return [line.strip() for line in fi]


def augment(sample, lookup_tables, aug_num):
    p = re.compile("\[(?P<word>.+?)\]\((?P<type>.+?)\)")
    new_samples = []
    for match in p.finditer(sample):
        type_ = match.group("type")
        word = match.group("word")
        choices = random.sample(lookup_tables[type_], k=aug_num)
        word in choices and choices.remove(word)
        for choice in choices:
            repstr = f"[{choice}]({type_})"
            new_sample = sample.replace(sample[match.start():match.end()], repstr)
            new_samples.append(new_sample)
    return new_samples


def generate_augment_data(source_path, save_path, lookup_paths, aug_num=3):
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


if __name__ == "__main__":
    lookup_paths = {
        "player": "../data/player.txt",
        "team": "../data/team.txt",
        "league": "../data/league.txt"
    }
    generate_augment_data("../data/test.md", "../data/aug.md", lookup_paths, 2)
