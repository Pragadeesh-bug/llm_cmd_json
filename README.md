# LLM Command-to-JSON Action Parser

## Files
- `command_to_actions.py`: local GGUF LLM integration, JSON parser, schema validation, demo parser, and 20 tests.
- `test_commands_and_outputs.txt`: all 20 test commands and their outputs.

## Run the 20 tests
```bash
python3 command_to_actions.py --tests
```

## Run with the local LLM
Install `llama-cpp-python`, then put a compatible GGUF instruct model at:
`models/qwen2.5-1.5b-instruct-q4_k_m.gguf`

```bash
python3 -m pip install llama-cpp-python
python3 command_to_actions.py "go to the door and wave"
```

You can change the model path:
```bash
MODEL_PATH=/path/to/model.gguf python3 command_to_actions.py "go to the door and wave"
```

## Run without a model
```bash
python3 command_to_actions.py --demo "go to the door and wave"
```

The parser checks JSON syntax, the top-level `actions` list, supported action names,
required fields, field types, and non-negative numeric values. The demo parser is
only for reproducible examples; natural-language interpretation in the actual LLM
mode is handled by the local model. The allowed actions should be matched to the
real robot's controller before execution.
