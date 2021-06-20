import random
import re


def load(path):
    with open(path, encoding="utf-8") as fi:
        return [line.strip() for line in fi]


def augment(sample, lookup_tables, aug_num):
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
        "player": "data/players.txt",
        "team": "data/teams.txt",
        "league": "data/leagues.txt"
    }
    # generate_augment_data("data/train.md", "data/aug_train.md", lookup_paths, 5)
    generate_augment_data("data/dev.md", "data/aug_dev.md", lookup_paths, 5)