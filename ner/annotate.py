import ahocorasick
from itertools import groupby
from operator import itemgetter


def load(path):
    with open(path, encoding="utf-8") as fi:
        return [line.strip() for line in fi]


def make_automation(lookup_paths):
    A = ahocorasick.Automaton()
    for type_, path in lookup_paths.items():
        for term in load(path):
            A.add_word(term, (term, type_))
    A.make_automaton()
    return A


def longest_match(matches):
    matches = [(match[0]-len(match[1][0])+1, match[1]) for match in matches]
    matches = sorted(matches, key=itemgetter(0))
    for pos, match_set in groupby(matches, itemgetter(0)):
        yield max(match_set, key=lambda x: len(x[1][0]))


def annotate(A, text):
    current = -1
    annotated_text = []
    for pos, match in longest_match(A.iter(text)):
        if pos <= current:
            continue
        annotated_text.extend(text[current+1:pos])
        annotation = f"[{match[0]}]({match[1]})"
        annotated_text.extend(annotation)
        current = pos + len(match[0]) - 1
    annotated_text.extend(text[current+1:])
    return "".join(annotated_text)


def annotate_file(source_path, save_path, lookup_paths):
    A = make_automation(lookup_paths)
    with open(source_path, encoding="utf-8") as fi, \
            open(save_path, "w", encoding="utf-8") as fo:
        for line in fi:
            new_line = annotate(A, line.strip())
            fo.write(new_line)
            fo.write("\n")

        
if __name__ == "__main__":
    lookup_paths = {
        "player": "../data/player.txt",
        "team": "../data/team.txt",
        "league": "../data/league.txt"
    }
    annotate_file("../data/test.txt", "test.md", lookup_paths)
