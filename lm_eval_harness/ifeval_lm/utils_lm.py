import re


def process_results(doc, results):
    (loglikelihood,) = results
    prompt = doc["prompt"]
    _words = len(re.split(r"\s+", prompt))
    _bytes = len(prompt.encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
