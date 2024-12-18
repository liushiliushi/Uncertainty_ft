# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from functools import partial

from llama_recipes.datasets2.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets2.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets2.custom_dataset import get_custom_dataset
from llama_recipes.datasets2.gsm8k_dataset import get_gsm8k_dataset2
from llama_recipes.datasets2.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets2.toxicchat_dataset import get_llamaguard_toxicchat_dataset as get_llamaguard_toxicchat_dataset
from llama_recipes.datasets2.professional_law import get_professional_law
from llama_recipes.datasets2.strategyqa_dataset import get_strategyqa
from llama_recipes.datasets2.object_cou import get_object_cou2
from llama_recipes.datasets2.trivia_qa import get_trivia_qa, get_trivia_qa_dynamic

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset2,
    "professional_law": get_professional_law,
    "strategyqa": get_strategyqa,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa
}

DATASET_PREPROC_dynamic = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "llamaguard_toxicchat_dataset": get_llamaguard_toxicchat_dataset,
    "gsm8k_dataset": get_gsm8k_dataset2,
    "professional_law": get_professional_law,
    "strategyqa": get_strategyqa,
    "object_cou": get_object_cou2,
    "trivia_qa": get_trivia_qa_dynamic
}