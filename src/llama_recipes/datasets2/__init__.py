# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from llama_recipes.datasets2.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets2.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets2.custom_dataset import get_custom_dataset
from llama_recipes.datasets2.gsm8k_dataset import get_gsm8k_dataset2, get_gsm8k_dataset_raw, get_gsm8k_dataset_yes, get_gsm8k_dataset_confidence, get_gsm8k_dataset_implicit
from llama_recipes.datasets2.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets2.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.datasets2.professional_law import get_professional_law
from llama_recipes.datasets2.strategyqa_dataset import get_strategyqa, get_strategyqa_yes, get_strategyqa_confidence
from llama_recipes.datasets2.object_cou import get_object_cou2
from llama_recipes.datasets2.trivia_qa import get_trivia_qa, get_trivia_qa_raw, get_trivia_qa_yes, get_trivia_qa_cons, get_trivia_qa_confidence
from llama_recipes.datasets2.truthful_qa import get_truthful_qa, get_truthful_qa_raw, get_truthful_qa_yes, get_truthful_qa_reflection, get_truthful_qa_confidence
from llama_recipes.datasets2.hotpot_qa import get_hotpot_qa, get_hotpot_qa_raw, get_hotpot_qa_yes, get_hotpot_qa_reflection, get_hotpot_qa_confidence
DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset2,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa,
    "truthful_qa": get_truthful_qa,
    "hotpot_qa": get_hotpot_qa,
}

DATASET_RAW = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_raw,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_raw,
    "truthful_qa": get_truthful_qa_raw,
    "hotpot_qa": get_hotpot_qa_raw,
}

DATASET_YES = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_yes,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa_yes,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_yes,
    "truthful_qa": get_truthful_qa_yes,
    "hotpot_qa": get_hotpot_qa_yes,
}

DATASET_REFLECTION = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_yes,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa_yes,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_yes,
    "truthful_qa": get_truthful_qa_reflection,
    "hotpot_qa": get_hotpot_qa_reflection,
}

DATASET_CONS = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_yes,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa_yes,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_cons,
    "truthful_qa": get_truthful_qa_yes,
    "hotpot_qa": get_hotpot_qa_yes,
}

DATASET_CONFIDENCE_ANSWER = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_confidence,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa_confidence,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_confidence,
    "truthful_qa": get_truthful_qa_confidence,
    "hotpot_qa": get_hotpot_qa_confidence,
}

DATASET_IMPLICIT = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset_implicit,
    "professional_law": get_professional_law,
    "strategy_qa": get_strategyqa_confidence,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_confidence,
    "truthful_qa": get_truthful_qa_confidence,
    "hotpot_qa": get_hotpot_qa_confidence,
}