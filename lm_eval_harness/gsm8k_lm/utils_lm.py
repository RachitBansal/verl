import re


def process_results(doc, results):
    """Process loglikelihood results for GSM8K perplexity.

    For output_type=loglikelihood, results is [(loglikelihood, is_greedy)].
    We compute perplexity over the answer tokens only
    (the question is context, not scored).
    """
    ll, _is_greedy = results[0]
    answer = doc["answer"]
    _words = len(re.split(r"\s+", answer))
    _bytes = len(answer.encode("utf-8"))
    return {
        "word_perplexity": (ll, _words),
        "byte_perplexity": (ll, _bytes),
        "bits_per_byte": (ll, _bytes),
    }
