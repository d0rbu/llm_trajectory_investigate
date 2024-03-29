import json
import os
from hashlib import md5
from typing import Iterable, Mapping


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


def get_prompts(
    type_hierarchy: Iterable[str] = [],
    return_hash: bool = False
) -> Iterable[Mapping]:
    # check to make sure it is a valid type hierarchy
    current_types = PROMPT_TYPES
    for prompt_type in type_hierarchy:
        assert current_types is not True, f"went past maximum depth prompt type: {prompt_type}, from {type_hierarchy}"
        assert prompt_type in current_types, f"unexpected prompt type: {prompt_type}, from {type_hierarchy}"
        current_types = current_types[prompt_type]

    current_prompts = full_prompts

    for type_level, current_type in enumerate(type_hierarchy):
        current_prompts = [  # filter down each time by the types in the hierarchy
            prompt for prompt in current_prompts
            if len(prompt["type"]) > type_level and prompt["type"][type_level] == current_type
        ]

    if return_hash:
        return current_prompts, int(md5(str.encode(json.dumps(current_prompts, sort_keys=True))).hexdigest(), 16)

    return current_prompts
