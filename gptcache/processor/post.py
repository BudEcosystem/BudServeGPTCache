import random
from typing import List, Any

import numpy

from gptcache.utils import softmax


def random_one(messages: List[Any], scores: List[float]) -> Any:
    """Randomly select one result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import random_one

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer, score = random_one(messages, scores)
    """
    m_s = list(zip(messages, scores))
    selected_index = random.choice(range(len(messages)))
    return m_s[selected_index]


def first(messages: List[Any], scores: List[float]) -> Any:
    """Get the first result after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]

    Example:
        .. code-block:: python

            from gptcache.processor.post import first

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer = first(messages, scores)
            assert answer = (messages[0], scores[0])
    """
    m_s = list(zip(messages, scores))
    return m_s[0]


def nop(messages: List[Any], scores: List[float]) -> Any:
    """No change after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to `messages`
    :type scores: List[float]

    Example:
        .. code-block:: python

            from gptcache.processor.post import nop

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer_scores = nop(messages, scores)
            assert answer_scores[0] = (messages[0], scores[0])
    """
    m_s = list(zip(messages, scores))
    return m_s


def temperature_softmax(messages: List[Any], scores: List[float], temperature: float = 0.0) -> Any:
    """Post processing with temperature softmax after evaluation.

    :param messages: A list of candidate outputs.
    :type messages: List[Any]
    :param scores: A list of evaluation scores corresponding to `messages`
    :type scores: List[float]
    :param temperature: A non-negative number of sampling temperature, defaults to 0.
                        A higher temperature makes the output more random.
                        A lower temperature means a more deterministic and confident output.
    :type temperature: float

    Example:
        .. code-block:: python

            from gptcache.processor.post import temperature_softmax

            messages = ["message 1", "message 2", "message 3"]
            scores = [0.9, 0.5, 0.1]
            answer = temperature_softmax(messages, scores, temperature=0.5)
    """
    m_s = list(zip(messages, scores))
    if temperature > 0:
        scores = softmax([x / temperature for x in scores])
        index_of_selected_message = numpy.random.choice(
            range(len(messages)), size=1, p=scores
        )[0]
        return m_s[index_of_selected_message][0]
    else:
        return sorted(m_s, key=lambda x: x[1], reverse=True)[0]
