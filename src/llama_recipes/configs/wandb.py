# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class wandb_config_qwen:
    project: str = 'grid_2000_qwen_coarse' # wandb project name
    entity: Optional[str] = None # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[bool] = False
    id: Optional[str] = None
    # allow_val_change: Optional[bool] = True
@dataclass
class wandb_config_mini:
    project: str = 'grid_2000_ministral_coarse' # wandb project name
    entity: Optional[str] = None # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[bool] = False
    id: Optional[str] = None
    # allow_val_change: Optional[bool] = True
@dataclass
class wandb_config_llama:
    project: str = 'grid_2000_llama_gpt2' # wandb project name
    entity: Optional[str] = None # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    resume: Optional[bool] = False
    id: Optional[str] = None