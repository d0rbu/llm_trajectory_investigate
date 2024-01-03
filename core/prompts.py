import json
import os
from typing import Iterable


PROMPT_TYPES = {
    "knowledge": {
        "capitals": True,
    },
    "reasoning": {
        "math": True,
    },
}
PROMPTS_PATH = os.path.join("core", "prompts.json")

with open(PROMPTS_PATH, "r", encoding="utf-8") as prompts_file:
    full_prompts = json.load(prompts_file)


def get_prompts(type_hierarchy: Iterable[str] = []):
    # check to make sure it is a valid type hierarchy
    current_types = PROMPT_TYPES
    for prompt_type in type_hierarchy:
        assert current_types is not True, f"went past maximum depth prompt type: {prompt_type}, from {type_hierarchy}"
        assert prompt_type in current_types, f"unexpected prompt type: {prompt_type}, from {type_hierarchy}"
        current_types = PROMPT_TYPES[prompt_type]

    current_prompts = full_prompts

    for type_level, current_type in enumerate(type_hierarchy):
        current_prompts = [  # filter down each time by the types in the hierarchy
            prompt for prompt in current_prompts
            if len(prompt["type"]) > type_level and prompt["type"][type_level] == current_type
        ]

    return current_prompts
