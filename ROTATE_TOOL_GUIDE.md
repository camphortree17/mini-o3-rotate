# Rotate Tool Modification Guide

This document describes the code changes for adding an initial image-rotation tool before the existing crop/grounding tool, and the data format expected for training or inference.

## Design

Rotation is treated as an initial image-orientation correction step. The model may call:

```text
<think>...</think><rotate>{"angle": 90}</rotate>
```

The rollout executes the rotation and treats the rotated image as the new `original_image` for subsequent crop calls. Therefore the later crop trajectory can keep using the original Mini-o3 format:

```text
<think>...</think><grounding>{"bbox_2d": [x0, y0, x1, y1], "source": "original_image"}</grounding>
```

The rotation angle is clockwise and must be one of:

```text
0, 90, 180, 270
```

`<rotate>` is only supported as the first tool action. Later tool actions should use `<grounding>`.

## Modified Code

### `verl/trainer/constants.py`

Added:

- `TOOL_ROTATE_CROP_SYSTEM_PROMPT`
- `SYSTEM_PROMPT_MAP["tool_rotate_crop"]`
- `TOOL_CALL_ROTATE_MULTI_TRUN_PROMPT`

The new system prompt tells the model to first decide whether rotation is needed, then use the usual crop tool after the rotated image has become the new `original_image`.

### `verl/workers/rollout/vllm_rollout/function_tools.py`

Added:

- `rotate_image(image, angle)`
- `prepare_rotate_inputs_multi_turn(json_objects, observations)`
- `parse_single_tool_json(response, tag)`

Also updated `get_valid_mask(...)` so `data.tool_call="rotate_crop"` is handled like the existing crop tool when computing tool-use statistics and valid masks.

### `verl/workers/rollout/vllm_rollout/vllm_async_engine.py`

This is the main path for README-style inference/training with:

```text
actor_rollout_ref.rollout.mode="async"
```

Updated behavior:

- Detects both `<rotate>...</rotate>` and `<grounding>...</grounding>`.
- Rejects responses that call both tools in one turn.
- Allows `<rotate>` only at `current_iteration == 0`.
- Executes rotation with PIL.
- Updates `observations_list[0]` to the rotated image.
- Updates `image_size_used_list[0]` to the processed rotated image size.
- Appends the rotated image to `multi_modal_data["image"]` for the next visual turn.
- After rotation, later crop calls with `"source": "original_image"` operate on the rotated image.

### `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

Added minimal sync-mode compatibility:

- Stop generation on either `</grounding>` or `</rotate>`.
- Parse `<rotate>` tool calls.
- Execute rotation as the initial tool.
- Treat the rotated image as the new `original_image` for later crop.

The async path is still the primary path used by the Mini-o3 command examples.

### Reward Format Files

Updated:

- `verl/utils/reward_score/general_qa_tool.py`
- `verl/utils/reward_score/general_qa_tool_mc.py`
- `verl/utils/reward_score/general_qa_tool_qwen.py`

The format reward now accepts multi-turn trajectories like:

```text
<think>...</think><rotate>{"angle": 90}</rotate>
<think>...</think><grounding>{"bbox_2d": [...], "source": "original_image"}</grounding>
<think>...</think><answer>...</answer>
```

It also rejects a single response that contains both `<rotate>` and `<grounding>`.

### `verl/trainer/ppo/ray_trainer.py`

Updated tool-call logging/statistics so `rotate_crop` is treated as a tool-call mode and `<rotate>` is counted as an attempted tool call.

### `verl/trainer/config/ppo_trainer.yaml`

Updated defaults:

```yaml
data:
  system_prompt: tool_rotate_crop
  tool_call: rotate_crop

actor_rollout_ref:
  rollout:
    max_generation_round: 6
    limit_mm_per_prompt: {'image': 12}
    val_max_generation_round: 12
    val_limit_mm_per_prompt: {'image': 12}
    stop: [ '</grounding>', '</rotate>' ]
```

If you use command-line overrides, make sure they are consistent with these values.

## Required Inference / RL Settings

For rotate + crop inference or RL, use the multi-turn tool-call rollout and the new prompt/tool mode:

```bash
data.system_prompt="tool_rotate_crop" \
data.tool_call="rotate_crop" \
actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
actor_rollout_ref.rollout.mode="async" \
actor_rollout_ref.rollout.max_generation_round=6 \
'actor_rollout_ref.rollout.limit_mm_per_prompt={"image": 12}' \
actor_rollout_ref.rollout.val_max_generation_round=12 \
'actor_rollout_ref.rollout.val_limit_mm_per_prompt={"image": 12}' \
actor_rollout_ref.rollout.use_raw_image=True \
'actor_rollout_ref.rollout.stop=["</grounding>","</rotate>"]'
```

Keep `use_raw_image=True` for this project’s crop tool path, because the rollout uses the original image for accurate crop/rotation execution while the model sees processed images.

## RL / Evaluation JSON Format

The RL and evaluation data format remains the same as the original Mini-o3 JSON format:

```json
[
  {
    "images": [
      "path/to/rotated_input.jpg"
    ],
    "doc_id": "sample_000001",
    "problem": "<image>\nQuestion: What is written on the sign?",
    "solution": "The answer text or option",
    "data_source": "visual_probe_train"
  }
]
```

Important notes:

- `problem` must contain `<image>`.
- `images` should contain the image that the model initially sees. For rotation training, this is usually the deliberately rotated image.
- The model decides the first tool call from this initial image.
- After `<rotate>`, later crops should still use `"source": "original_image"`.

## Inference Trajectory Output Format

When running inference with:

```bash
actor_rollout_ref.rollout.save_traj=True \
actor_rollout_ref.rollout.save_traj_dir=${DEMO_OUTPUT_DIR}
```

the rollout saves one directory per generated trajectory:

```text
${DEMO_OUTPUT_DIR}/
  sample_000001/
    traj.jsonl
    original_image.jpg
    resized_image.jpg
    observation_1_original_image.jpg
    observation_1_resized_image.jpg
    observation_2_original_image.jpg
    observation_2_resized_image.jpg
```

If `rollout.n > 1`, the directory name is suffixed by rollout index, for example:

```text
sample_000001_0/
sample_000001_1/
```

### `traj.jsonl`

`traj.jsonl` is JSON Lines. The first line records the input metadata:

```json
{
  "doc_id": "sample_000001",
  "initial_prompt": "<image>\nQuestion: What is written on the sign?",
  "ground_truth": "..."
}
```

Each model turn is then appended as one line:

```json
{
  "turn_idx": 0,
  "text_output": "<think>The image is sideways, so I should rotate it upright.</think><rotate>{\"angle\": 90}</rotate>"
}
```

```json
{
  "turn_idx": 1,
  "text_output": "<think>Now I can inspect the sign. I need to zoom in.</think><grounding>{\"bbox_2d\": [0.32, 0.18, 0.71, 0.39], \"source\": \"original_image\"}</grounding>"
}
```

```json
{
  "turn_idx": 2,
  "text_output": "<think>The zoomed image is readable.</think><answer>...</answer>"
}
```

If a tool call fails, an additional line can be written:

```json
{
  "error_info": "<|im_end|>\n<|im_start|>user\nERROR occurs during grounding. Error Information: ...\nPlease analyze the error information obtained from the function tool and adjust your response..."
}
```

### Saved Images

The image files are saved by `save_traj_and_obs(...)`:

- `original_image.jpg`: the initial raw image loaded from disk.
- `resized_image.jpg`: the processed image that is fed to the model in the initial prompt.
- `observation_1_original_image.jpg`: the raw tool output from Action 0.
- `observation_1_resized_image.jpg`: the processed version of Action 0 output fed back to the model.
- `observation_2_original_image.jpg`: the raw tool output from Action 1.
- `observation_2_resized_image.jpg`: the processed version of Action 1 output fed back to the model.

For rotate-crop trajectories, if turn 0 is `<rotate>`, then:

- `observation_1_original_image.jpg` is the rotated/corrected full image.
- `observation_1_resized_image.jpg` is the processed rotated image shown to the model.
- Later `<grounding>` calls with `"source": "original_image"` are executed against the rotated/corrected full image internally.

The saved image filenames still use the generic `observation_i` naming because they represent tool outputs shown back to the model. This does not change the tool-coordinate rule: after rotation, crop source `"original_image"` refers to the corrected image inside the rollout state.

### Converting Inference Output to SFT Data

To turn an inference trajectory into ShareGPT-style SFT data:

1. Read `traj.jsonl`.
2. Use the first line's `initial_prompt` as the first human message.
3. Use each `text_output` line as an assistant message.
4. Insert a human observation message after every tool-call assistant message.
5. Reference the corresponding saved observation image in the `images` list, in the same order that `<image>` appears in the conversation.

For a rotate-crop trajectory, the SFT sample should look like:

```json
{
  "images": [
    "sample_000001/original_image.jpg",
    "sample_000001/observation_1_resized_image.jpg",
    "sample_000001/observation_2_resized_image.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nQuestion: What is written on the sign?"
    },
    {
      "from": "gpt",
      "value": "<think>The image is sideways, so I should rotate it upright.</think><rotate>{\"angle\": 90}</rotate>"
    },
    {
      "from": "human",
      "value": "After the above Action 0, the image has been rotated and set as the new original image:\n<image>.\nContinue your reasoning process inside <think> and </think>. If needed, zoom in on the current original image or any observation by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."
    },
    {
      "from": "gpt",
      "value": "<think>Now I can inspect the sign. I need to zoom in.</think><grounding>{\"bbox_2d\": [0.32, 0.18, 0.71, 0.39], \"source\": \"original_image\"}</grounding>"
    },
    {
      "from": "human",
      "value": "After the above Action 1, here is the the zoom-in image (Observation 1):\n<image>.\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."
    },
    {
      "from": "gpt",
      "value": "<think>The zoomed image is readable.</think><answer>...</answer>"
    }
  ]
}
```

For LLaMA-Factory SFT, the `images` list must contain exactly the image files corresponding to the `<image>` placeholders in conversation order. If you only train on the initial image and text-only tool traces, keep only one `<image>` and one image path; if you train the full multi-turn visual trajectory, include the observation images as shown above.

## SFT / Cold-Start Conversation Format

For supervised data, use the existing ShareGPT-style format:

```json
{
  "images": [
    "path/to/rotated_input.jpg"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nQuestion: What is written on the sign?"
    },
    {
      "from": "gpt",
      "value": "<think>The image is sideways, so I should rotate it upright before reading the sign.</think><rotate>{\"angle\": 90}</rotate>"
    },
    {
      "from": "human",
      "value": "After the above Action 0, the image has been rotated and set as the new original image:\n<image>.\nContinue your reasoning process inside <think> and </think>. If needed, zoom in on the current original image by outputting <grounding> and </grounding>."
    },
    {
      "from": "gpt",
      "value": "<think>The text is still small, so I need to zoom in on the sign area.</think><grounding>{\"bbox_2d\": [0.32, 0.18, 0.71, 0.39], \"source\": \"original_image\"}</grounding>"
    },
    {
      "from": "human",
      "value": "After the above Action 1, here is the the zoom-in image (Observation 1):\n<image>.\nContinue your reasoning process inside <think> and </think>."
    },
    {
      "from": "gpt",
      "value": "<think>The zoomed image is clear enough to read the sign.</think><answer>...</answer>"
    }
  ]
}
```

For LLaMA-Factory, keep the dataset mapping style from the original README:

```json
{
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```

## Recommended Data Construction Pipeline

Recommended pipeline for building rotate-tool training data:

1. Start from an upright image and question.
2. Rotate the image by a known angle to create `rotated_input.jpg`.
3. Ask a stronger model to generate the first rotate reasoning step for `rotated_input.jpg`.
4. Execute the inverse correction angle through `<rotate>{"angle": ...}</rotate>`.
5. Treat the corrected image as the new `original_image`.
6. Generate or reuse crop trajectories on the corrected image.
7. Keep all later crop calls as `"source": "original_image"`.

This avoids rewriting crop sources to `observation_i`.

## Coordinate Rule

After the first rotate step, all crop boxes must be expressed in the coordinate system of the rotated/corrected image.

Because the code treats the corrected image as the new `original_image`, do not use crop boxes from the wrongly rotated input unless they have been transformed into the corrected image coordinate system.

## Validation Performed

The following lightweight checks were run:

- Python AST parsing for all modified Python files.
- PIL rotation smoke test.
- `<rotate>` JSON parsing smoke test.
- Format reward smoke test for:
  - valid `rotate -> crop -> answer`
  - invalid same-turn `rotate + crop`

Full vLLM rollout was not run in this local environment because the complete training/runtime dependencies are not installed here.
