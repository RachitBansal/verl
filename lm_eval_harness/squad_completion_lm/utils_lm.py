import re


def process_results(doc, results):
    ll, _is_greedy = results[0]
    answer = doc["value"]
    _words = len(re.split(r"\s+", answer))
    _bytes = len(answer.encode("utf-8"))
    return {
        "word_perplexity": (ll, _words),
        "byte_perplexity": (ll, _bytes),
        "bits_per_byte": (ll, _bytes),
    }
