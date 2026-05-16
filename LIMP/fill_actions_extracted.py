from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from model_backend import get_default_model
from runtime_hparams import get_enable_thinking_default


FILES_DIR = Path(__file__).resolve().parent.parent / "Files"
PROMPT_FILE = FILES_DIR / "actions_extracted.json"
VIDEOS_DIR = Path("/data/LPP/cvpr/muti_agent/MUMA-TOM-BENCHMARK/videos")


def load_json(path: Path) -> Any:
    with path.open("r") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def sample_video_frames(video_path: Path, num_frames: int) -> tuple[list[Image.Image], list[int], list[float]]:
    from decord import VideoReader

    video = VideoReader(str(video_path))
    frame_total = len(video)
    if frame_total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    sample_count = max(1, min(num_frames, frame_total))
    denominator = max(sample_count - 1, 1)
    raw_indices = [round(position * (frame_total - 1) / denominator) for position in range(sample_count)]
    frame_indices = list(dict.fromkeys(int(index) for index in raw_indices))

    try:
        fps = float(video.get_avg_fps())
    except Exception:
        fps = 0.0

    frames: list[Image.Image] = []
    timestamps: list[float] = []
    for frame_index in frame_indices:
        frame_array = video[frame_index].asnumpy()
        frames.append(Image.fromarray(frame_array))
        timestamps.append(frame_index / fps if fps > 0 else -1.0)

    return frames, frame_indices, timestamps


def build_instruction(prompt_text: str, frame_indices: list[int], timestamps: list[float]) -> str:
    frame_lines: list[str] = []
    for order, (frame_index, timestamp) in enumerate(zip(frame_indices, timestamps), start=1):
        if timestamp >= 0:
            frame_lines.append(f"{order}. frame index {frame_index}, timestamp about {timestamp:.2f}s")
        else:
            frame_lines.append(f"{order}. frame index {frame_index}")

    return "\n\n".join(
        [
            prompt_text,
            (
                "The following frames are shown in chronological order from the same video. "
                "Infer the action sequence using only evidence that is visually supported by the frames."
            ),
            "Chronological frame references:\n" + "\n".join(frame_lines),
            "Output only the requested single-line action description. Do not use bullet points, numbering, markdown, or JSON.",
        ]
    )


def extract_action_text(prompt_text: str, video_path: Path, num_frames: int, max_new_tokens: int) -> dict[str, Any]:
    model = get_default_model()
    frames, frame_indices, timestamps = sample_video_frames(video_path, num_frames=num_frames)
    instruction = build_instruction(prompt_text, frame_indices, timestamps)

    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    content.extend({"type": "image", "image": frame} for frame in frames)

    action_text = model.chat(
        [
            {
                "role": "system",
                "content": (
                    "You analyze chronological household video frames and describe the visible actions. "
                    "Follow the user's output format exactly."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enable_thinking=get_enable_thinking_default(),
    )
    action_text = " ".join(action_text.split())

    return {
        "action": action_text,
        "sampled_frame_indices": frame_indices,
        "sampled_timestamps": timestamps,
        "video_path": str(video_path),
    }


def structure_action_text(raw_action_text: str, max_new_tokens: int = 256) -> dict[str, list[str] | str]:
    if not raw_action_text:
        return {"action_summary": "", "actions_list": [], "observations": []}

    model = get_default_model()
    data = model.generate_json(
        [
            {
                "role": "system",
                "content": "You convert action descriptions into structured JSON. Return JSON only.",
            },
            {
                "role": "user",
                "content": (
                    "Convert the following single-line video action description into valid JSON with this schema:\n"
                    "{\n"
                    "  \"action_summary\": \"single-line chronological action summary\",\n"
                    "  \"actions_list\": [\"...\"],\n"
                    "  \"observations\": [\"...\"]\n"
                    "}\n"
                    "Rules:\n"
                    "- Keep actions_list chronological.\n"
                    "- Keep action_summary faithful to the input.\n"
                    "- Use short action phrases in actions_list.\n"
                    "- Put only direct visible facts in observations.\n"
                    "- Do not invent details not supported by the input.\n\n"
                    f"Input: {raw_action_text}"
                ),
            },
        ],
        max_new_tokens=max_new_tokens,
        enable_thinking=get_enable_thinking_default(),
    )

    if not isinstance(data, dict):
        return {"action_summary": raw_action_text, "actions_list": [], "observations": []}

    action_summary = str(data.get("action_summary", "")).strip() or raw_action_text
    actions_list = [str(item).strip() for item in data.get("actions_list", []) if str(item).strip()]
    observations = [str(item).strip() for item in data.get("observations", []) if str(item).strip()]
    return {
        "action_summary": action_summary,
        "actions_list": actions_list,
        "observations": observations,
    }


def fill_actions_extracted(
    episode_list: list[int],
    num_frames: int = 16,
    overwrite_existing: bool = False,
    max_new_tokens: int = 384,
) -> None:
    data = load_json(PROMPT_FILE)

    for episode_id in episode_list:
        episode_key = str(episode_id)
        if episode_key not in data:
            print(f"Skip episode {episode_id}: missing entry in {PROMPT_FILE}")
            continue

        entry = data[episode_key]
        if not isinstance(entry, dict):
            print(f"Skip episode {episode_id}: invalid JSON entry")
            continue

        existing_action = str(entry.get("action", "")).strip()
        if existing_action and not overwrite_existing:
            print(f"Skip episode {episode_id}: action already exists")
            continue

        prompt_text = str(entry.get("prompt", "")).strip()
        if not prompt_text:
            print(f"Skip episode {episode_id}: missing prompt")
            continue

        video_path = VIDEOS_DIR / f"video_{episode_id}.mp4"
        if not video_path.exists():
            print(f"Skip episode {episode_id}: missing video {video_path}")
            continue

        print(f"Extract visual action for episode {episode_id}")
        raw_result = extract_action_text(
            prompt_text=prompt_text,
            video_path=video_path,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
        )
        structured = structure_action_text(raw_result["action"])

        entry["action"] = raw_result["action"]
        entry["actions_list"] = structured["actions_list"]
        entry["observations"] = structured["observations"]
        entry["sampled_frame_indices"] = raw_result["sampled_frame_indices"]
        entry["sampled_timestamps"] = raw_result["sampled_timestamps"]
        entry["video_path"] = raw_result["video_path"]
        save_json(PROMPT_FILE, data)

        print(raw_result["action"])


if __name__ == "__main__":
    episode_list = [4005, 4009, 4017, 4018, 4023, 4034, 4037, 4041, 4043, 4054, 4057, 4059, 4063, 4070, 4077, 4078, 4081, 4083, 4098, 4103, 4105, 4106, 4124, 4145, 4150, 4162, 4172, 4184, 4190, 4198, 4200, 4284, 4324, 4327, 4331, 4338, 4343, 4367, 4369, 4370, 4372, 4374, 4385, 4416, 4419, 4423, 4429, 4439, 4441, 4449, 4452, 4453, 4469, 4473, 4482, 4485, 4487, 4488, 4490, 4499, 4505, 4506, 4510, 4512, 4520, 4525, 4540, 4542, 4546, 4552, 4556, 4559, 4567, 4568, 4594, 4604, 4606, 4618, 4623, 4641, 4656, 4658, 5010, 5017, 5039, 5042, 5068, 5080, 5084, 5091, 5095, 5099, 5103, 5105, 5121, 5138, 5154, 5165, 5173, 5175, 5197, 5302, 5379, 5381, 5509, 4047, 4101, 4102, 4113, 4117, 4123, 4133, 4140, 4160, 4173, 4176, 4178, 4280, 4285, 4312, 4328, 4332, 4365, 4415, 4458, 4463, 4526, 4527, 4529, 4551, 4560, 4576, 4584, 4621, 4667, 5014, 5049, 5082, 5093, 5098, 5123, 5126, 5127, 4455, 4375, 4164, 4224, 4329, 4575, 5163, 135, 138, 153, 193, 389, 532, 538, 577, 628, 630, 642, 647, 766, 848, 865, 895, 910, 956, 1811, 1856, 2559, 3050, 3058, 3068, 3315, 129, 152, 161, 225, 263, 397, 578, 583, 601, 609, 640, 644, 682, 784, 801, 824, 857, 913, 1131, 1817, 1819, 2462, 3077, 42, 128, 130, 144, 154, 223, 393, 404, 528, 541, 548, 549, 557, 634, 790, 871, 905, 1758, 1818, 2053, 2070, 3074, 3092, 3098, 3129, 3130, 3308]

    # episode_list = [4005, 4009, 4017, 4018, 4023, 4034, 4037, 4041, 4043]
    num_frames = 24
    overwrite_existing = False

    fill_actions_extracted(
        episode_list=episode_list,
        num_frames=num_frames,
        overwrite_existing=overwrite_existing,
    )
