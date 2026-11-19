"""Build AgentV-RL SFT data with independently configurable solver and verifier backends."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
from hashlib import sha256
from itertools import islice
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
import unicodedata

from agentflow.utils.tag_util import find_tags


SOURCES = {
    "polaris": ("POLARIS-Project/Polaris-Dataset-53K", "polaris-data-53K.jsonl"),
    "deepscaler": ("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json"),
    "boba": ("inclusionAI/AReaL-boba-Data", "AReaL-boba-106k.jsonl"),
}
OUTPUT_FILES = {"train": "sft_multiturn_train.parquet", "validation": "sft_multiturn_test.parquet"}
SAMPLES_PER_PROBLEM = 8
DEFAULT_SOLVER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen3-4B"
BOXED_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
LOGGER = logging.getLogger(__name__)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def normalize_problem(source, row):
    if source == "boba":
        if row["task"] != "math":
            return None
        question = row["prompt"].strip().removeprefix("<｜User｜>").split("<｜Assistant｜>", 1)[0]
        question = question.rstrip().removesuffix(BOXED_INSTRUCTION)
        references = row["solutions"]
    else:
        question = row["problem"]
        references = [row["answer"]]

    question = question.strip()
    references = [str(answer).strip() for answer in references if answer is not None and str(answer).strip()]
    if not question or not references:
        return None
    canonical = unicodedata.normalize("NFC", " ".join(question.split()))
    return {
        "problem_id": sha256(canonical.encode("utf-8")).hexdigest(),
        "source": source,
        "question": question,
        "references": references,
    }


def load_problems(args):
    from huggingface_hub import hf_hub_download

    problems = {}
    for source, (repo_id, filename) in SOURCES.items():
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
        rows = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        usable = 0
        for row in rows:
            problem = normalize_problem(source, row)
            if problem is not None and problem["problem_id"] not in problems:
                problems[problem["problem_id"]] = problem
                usable += 1
                if usable == args.max_problems_per_source:
                    break
        LOGGER.info("%s: %d new math problems", source, usable)
    shuffled = list(problems.values())
    random.Random(args.seed).shuffle(shuffled)
    if not shuffled:
        raise ValueError("No math problems with reference answers were loaded")
    return shuffled


def label_candidate(completion, references, grade_answer):
    grades = []
    for reference in references:
        grade = grade_answer(completion.text, reference)
        grades.append(grade)
        if grade["correct"]:
            break
    grade = next((grade for grade in grades if grade["correct"]), grades[0])
    return {
        "candidate_id": completion.index,
        "solution": completion.text,
        "finish_reason": completion.finish_reason,
        **grade,
    }


def build_candidate_sampler(args):
    if args.solver_backend == "openai":
        from agentflow.backend.openai import OpenaiBackend
        from agentflow.config import load_config

        config = load_config(str(Path(__file__).resolve().parents[1] / "config" / "default.yml"))
        config["backend"]["model_path"] = args.solver_model
        config["backend"]["openai"].update(
            api_key=os.environ["SOLVER_OPENAI_API_KEY"],
            url=os.environ["SOLVER_OPENAI_BASE_URL"],
            max_concurrency=args.solver_api_max_concurrency,
        )
        request_options = {args.solver_api_token_limit_field: args.solver_max_tokens}
        choices_per_request = args.solver_api_choices_per_request
        if choices_per_request > 1:
            request_options["n"] = choices_per_request
        if args.solver_api_temperature is not None:
            request_options["temperature"] = args.solver_api_temperature
        backend = OpenaiBackend(config, logger=LOGGER, request_options=request_options)
        probe_pending = choices_per_request > 1

        def sample_with_api(chats, on_completion):
            nonlocal choices_per_request, probe_pending
            initial_metadata = []
            if probe_pending:
                # Probe one prompt so an unsupported n does not fail an entire concurrent batch.
                probe_pending = False
                _, (probe,) = backend.generate(chats[:1], on_completion=on_completion)
                response = probe.get("raw")
                rejected_n = (
                    probe.get("error_status") in {400, 422}
                    and "n must be 1" in probe.get("error_message", "").casefold()
                )
                if rejected_n or (response is not None and len(response.choices) == 1):
                    choices_per_request = 1
                    backend.request_options.pop("n")
                    LOGGER.warning("Solver endpoint only provides one choice per request. Continuing with concurrent single-choice requests.")
                elif response is None:
                    raise RuntimeError("Candidate solver API probe failed. See the API backend logs.")
                if response is not None:
                    initial_metadata.append(probe)

            requests_per_problem = SAMPLES_PER_PROBLEM // choices_per_request
            prompts = [chat for chat in chats for _ in range(requests_per_problem)]
            pending_prompts = prompts[len(initial_metadata):]
            metadata = initial_metadata
            if pending_prompts:
                _, remaining_metadata = backend.generate(pending_prompts, on_completion=on_completion)
                metadata.extend(remaining_metadata)
            completions = []
            for _, meta in zip(prompts, metadata, strict=True):
                response = meta.get("raw")
                if response is None:
                    raise RuntimeError("Candidate solver API request failed after retries. See the API backend logs.")
                if len(response.choices) != choices_per_request:
                    raise RuntimeError(
                        f"Solver API returned {len(response.choices)} choices, expected {choices_per_request}. "
                        "Use a supported --solver-api-choices-per-request value."
                    )
                for choice in response.choices:
                    completions.append(SimpleNamespace(
                        index=len(completions) % SAMPLES_PER_PROBLEM,
                        text=choice.message.content or "",
                        finish_reason=choice.finish_reason,
                    ))
            return batches(completions, SAMPLES_PER_PROBLEM)

        return sample_with_api

    from vllm import LLM, SamplingParams

    # vLLM accepts a local model directory or downloads a Hub ID into the HF cache.
    model = LLM(
        model=args.solver_model or DEFAULT_SOLVER_MODEL,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.model_max_length,
        seed=args.seed,
    )
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(
        n=SAMPLES_PER_PROBLEM, temperature=1.0, top_k=50, max_tokens=args.solver_max_tokens
    )

    def sample_locally(chats, on_completion):
        prompts = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
        outputs = model.generate(prompts, sampling_params=sampling, use_tqdm=False)
        on_completion(sum(len(output.outputs) for output in outputs))
        return (output.outputs for output in outputs)

    return sample_locally


def generate_candidates(args):
    from tqdm import tqdm

    from agentflow.utils.math.answer_parser import grade_answer_verl

    problems = load_problems(args)
    sample = build_candidate_sampler(args)

    with TemporaryDirectory(dir=args.output_dir) as temp_dir:
        temporary = Path(temp_dir) / "candidates.jsonl"
        with temporary.open("w", encoding="utf-8") as stream, tqdm(
            total=len(problems) * SAMPLES_PER_PROBLEM, desc="Candidate generation", unit="solution",
            mininterval=2, miniters=1,
        ) as progress:
            labeled = 0
            for batch in batches(problems, args.solver_batch_size):
                chats = [
                    [{"role": "user", "content": f'{problem["question"]}\n\n{BOXED_INSTRUCTION}'}]
                    for problem in batch
                ]
                for problem, completions in zip(batch, sample(chats, progress.update), strict=True):
                    if len(completions) != SAMPLES_PER_PROBLEM:
                        raise RuntimeError(f'Expected eight outputs for {problem["problem_id"]}')
                    candidates = [
                        label_candidate(completion, problem["references"], grade_answer_verl)
                        for completion in completions
                    ]
                    stream.write(json.dumps({**problem, "candidates": candidates}, ensure_ascii=False) + "\n")
                labeled += len(batch)
                progress.set_postfix_str(f"labeled problems {labeled:,}/{len(problems):,}")
        temporary.replace(args.output_dir / "candidates.jsonl")
    LOGGER.info("Saved candidates for %d problems", len(problems))


def mixed_gradeable_problems(path, stats):
    for problem in read_jsonl(path):
        candidates = [
            candidate for candidate in problem["candidates"]
            if candidate["finish_reason"] == "stop" and candidate["parsed_pred"] and candidate["parsed_gt"]
        ]
        if len(candidates) != SAMPLES_PER_PROBLEM:
            stats["incomplete_or_ungradeable_problems"] += 1
        elif len({candidate["correct"] for candidate in candidates}) != 2:
            stats["uniform_label_problems"] += 1
        else:
            yield problem


def parse_verdict(text):
    answers = find_tags(text, ["answer"])
    verdict = answers[-1].body.strip().lower() if answers else ""
    return verdict if verdict in {"true", "false"} else None


def training_row(problem, candidate, direction, verifier, trace, tokenizer, args, stats):
    if not trace or trace[-1].role != "assistant":
        stats["missing_verdict"] += 1
        return None
    verdict = parse_verdict(trace[-1].content)
    if verdict is None:
        stats["missing_verdict"] += 1
        return None
    if (verdict == "true") != candidate["correct"]:
        stats["verdict_mismatch"] += 1
        return None

    stage = "planning"
    completed_stages = set()
    messages = []
    for message in trace:
        if message.role == "user" and message.content == verifier.DEFAULT_USER_STAGE_SUBTASK_BEGIN:
            stage = "validation"
        elif message.role == "user" and message.content == verifier.DEFAULT_USER_STAGE_REVIEW_MIDDLE:
            stage = "review"
        elif message.role == "assistant" and message.content.strip():
            completed_stages.add(stage)
        messages.append({"role": message.role, "content": message.content})
    if completed_stages != {"planning", "validation", "review"}:
        stats["incomplete_traces"] += 1
        return None

    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, enable_thinking=args.enable_thinking
    )
    if len(tokens) > args.max_sft_tokens:
        stats["overlength_traces"] += 1
        return None

    stats["kept_traces"] += 1
    return {
        "problem_id": problem["problem_id"],
        "source": problem["source"],
        "candidate_id": candidate["candidate_id"],
        "direction": direction,
        "correct": candidate["correct"],
        "enable_thinking": args.enable_thinking,
        "messages": messages,
    }


def verify_batch(problems, verifiers, tokenizer, args, stats):
    pairs = [(index, candidate) for index, problem in enumerate(problems) for candidate in problem["candidates"]]
    questions = [problems[index]["question"] for index, _ in pairs]
    solutions = [candidate["solution"] for _, candidate in pairs]
    rows_by_problem = [[] for _ in problems]
    for direction, verifier in verifiers.items():
        traces, _ = verifier.generate(questions, solutions)
        for (index, candidate), trace in zip(pairs, traces, strict=True):
            row = training_row(problems[index], candidate, direction, verifier, trace, tokenizer, args, stats)
            if row is not None:
                rows_by_problem[index].append(row)
    return rows_by_problem


def verify_api_batch(problems, verifier_factories, tokenizer, args, stats, executor, api_progress, trajectory_progress):
    cancelled = Event()
    progress_lock = Lock()

    def report_response(count):
        with progress_lock:
            api_progress.update(count)

    def run_trajectory(problem, candidate, direction):
        verifier = verifier_factories[direction]()

        def report_stage(stage):
            with progress_lock:
                api_progress.set_postfix_str(f"last stage started: {direction}/{stage}")

        (trace,), _ = verifier.generate(
            [problem["question"]], [candidate["solution"]],
            on_completion=report_response, on_stage=report_stage, cancel_event=cancelled,
        )
        return verifier, trace

    jobs = [
        (index, candidate, direction)
        for index, problem in enumerate(problems)
        for candidate in problem["candidates"]
        for direction in verifier_factories
    ]
    results = [None] * len(jobs)
    futures = {}
    try:
        for job_index, (problem_index, candidate, direction) in enumerate(jobs):
            future = executor.submit(run_trajectory, problems[problem_index], candidate, direction)
            futures[future] = job_index
        for future in as_completed(futures):
            job_index = futures[future]
            problem_index, candidate, direction = jobs[job_index]
            verifier, trace = future.result()
            results[job_index] = training_row(problems[problem_index], candidate, direction, verifier, trace,
                                              tokenizer, args, stats)
            trajectory_progress.set_postfix_str(f"retained {stats['kept_traces']:,}", refresh=False)
            trajectory_progress.update(1)
    finally:
        cancelled.set()
        for future in futures:
            future.cancel()

    rows_by_problem = [[] for _ in problems]
    for (index, _, _), row in zip(jobs, results, strict=True):
        if row is not None:
            rows_by_problem[index].append(row)
    return rows_by_problem


def write_parquet(args, verifier_factories, tokenizer, candidates_path):
    import pyarrow as pa
    from pyarrow import parquet as pq
    from tqdm import tqdm

    schema = pa.schema([
        ("problem_id", pa.string()),
        ("source", pa.string()),
        ("candidate_id", pa.int32()),
        ("direction", pa.string()),
        ("correct", pa.bool_()),
        ("enable_thinking", pa.bool_()),
        ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ])
    stats = Counter()
    split_sizes = Counter()
    using_api = args.verifier_backend == "openai"
    executor_context = ThreadPoolExecutor(max_workers=args.api_max_concurrency) if using_api else nullcontext()
    verifiers = {direction: factory() for direction, factory in verifier_factories.items()} if not using_api else None
    with TemporaryDirectory(dir=args.output_dir) as temp_dir:
        paths = {split: Path(temp_dir) / name for split, name in OUTPUT_FILES.items()}
        with (
            pq.ParquetWriter(paths["train"], schema) as train,
            pq.ParquetWriter(paths["validation"], schema) as val,
            tqdm(desc="Verifier problems", unit="problem", mininterval=2, miniters=1) as progress,
            tqdm(desc="Verifier API responses", unit="response", mininterval=2, miniters=1,
                 disable=not using_api) as api_progress,
            tqdm(desc="Verifier trajectories", unit="trajectory", mininterval=2, miniters=1,
                 disable=not using_api) as trajectory_progress,
            executor_context as executor,
        ):
            groups = mixed_gradeable_problems(candidates_path, stats)
            for problems in batches(groups, args.verifier_batch_size):
                if using_api:
                    verified_rows = verify_api_batch(problems, verifier_factories, tokenizer, args, stats,
                                                     executor, api_progress, trajectory_progress)
                else:
                    verified_rows = verify_batch(problems, verifiers, tokenizer, args, stats)
                rows_by_split = {"train": [], "validation": []}
                for rows in verified_rows:
                    if not rows:
                        continue
                    seen = split_sizes.total()
                    # Assign the whole problem together in the seeded candidate order.
                    val_target = max(1, round((seen + 1) * args.validation_fraction))
                    split = "validation" if seen and split_sizes["validation"] < val_target else "train"
                    split_sizes[split] += 1
                    rows_by_split[split].extend(rows)
                for split, writer in (("train", train), ("validation", val)):
                    if rows_by_split[split]:
                        writer.write_table(pa.Table.from_pylist(rows_by_split[split], schema=schema))
                progress.set_postfix_str(f"retained traces {stats['kept_traces']:,}", refresh=False)
                progress.update(len(problems))

        LOGGER.info("Synthesis totals: %s; unique problems by split: %s", dict(stats), dict(split_sizes))
        if len(split_sizes) != 2:
            raise ValueError("At least two distinct problems with retained traces are needed for train and validation")
        for split, path in paths.items():
            path.replace(args.output_dir / OUTPUT_FILES[split])


def generate_trajectories(args):
    candidates_path = args.output_dir / "candidates.jsonl"
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Missing {candidates_path}. Run --stage candidates first.")

    import ray
    from agentflow.agent.plan import BackwardVerifyAgent, MultiturnPlanSubtaskAgent
    from agentflow.config import load_config
    from agentflow.tools.code.python_execution_ray import PythonExecutionToolRay, create_python_actor
    from agentflow.tools.registry import ToolRegistry

    config = load_config(str(Path(__file__).resolve().parents[1] / "config" / "default.yml"))
    config["logging"]["level"] = "WARNING"
    config["backend"]["model_path"] = args.verifier_model or DEFAULT_VERIFIER_MODEL
    config["backend"]["sampling"]["max_tokens"] = args.verifier_max_tokens
    config["backend"]["vllm"].update(
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.model_max_length,
        max_num_batched_tokens=args.model_max_length,
        enable_sleep_mode=False,
    )
    ray.init(include_dashboard=False)
    try:
        if args.verifier_backend == "openai":
            from agentflow.backend.openai import OpenaiBackend

            config["backend"]["openai"].update(
                api_key=os.environ["OPENAI_API_KEY"],
                url=args.api_base_url or os.environ.get("OPENAI_BASE_URL"),
                max_concurrency=args.api_max_concurrency,
            )
            backend = OpenaiBackend(
                config,
                logger=LOGGER,
                request_options={args.api_token_limit_field: args.verifier_max_tokens},
                tool_observations_as_user=True,
            )
        else:
            from agentflow.backend.vllm import VllmBackend

            backend = VllmBackend(config, logger=LOGGER)
            backend.set_chat_template_defaults(enable_thinking=args.enable_thinking)

        if args.sft_tokenizer or args.verifier_backend == "openai":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.sft_tokenizer or DEFAULT_VERIFIER_MODEL)
        else:
            tokenizer = backend.tokenizer

        registry = ToolRegistry()
        actor = create_python_actor(time_limit_s=10, mem_limit_mb=512, max_task_retries=2)
        tool_type = PythonExecutionToolRay
        if args.verifier_backend == "openai":
            tool_lock = Lock()

            class SerialPythonTool(PythonExecutionToolRay):
                def run_batch(self, calls, **kwargs):
                    # The shared executor can replace its Ray actor after a timeout.
                    with tool_lock:
                        return super().run_batch(calls, **kwargs)

            tool_type = SerialPythonTool
        registry.register(tool_type(actor=actor, max_rounds=3))
        verifier_factories = {
            "forward": partial(MultiturnPlanSubtaskAgent, backend=backend, tool_registry=registry),
            "backward": partial(BackwardVerifyAgent, backend=backend, tool_registry=registry),
        }
        write_parquet(args, verifier_factories, tokenizer, candidates_path)
    finally:
        ray.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("all", "candidates", "verify"), default="all")
    parser.add_argument("--solver-backend", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--solver-model", help="API model ID, or local/HF model (vLLM default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--solver-api-max-concurrency", type=int, default=3)
    parser.add_argument("--solver-api-choices-per-request", type=int, choices=(1, 2, 4, 8), default=1,
                        help="Candidate solutions per API request, if the provider supports n")
    parser.add_argument("--solver-api-token-limit-field", choices=("max_completion_tokens", "max_tokens"),
                        default="max_completion_tokens")
    parser.add_argument("--solver-api-temperature", type=float, help="Optional temperature sent to the solver API")
    parser.add_argument("--verifier-backend", choices=("vllm", "openai"), default="vllm")
    parser.add_argument("--verifier-model", help="API model ID, or local/HF model (vLLM default: Qwen/Qwen3-4B)")
    parser.add_argument("--sft-tokenizer", help="Student tokenizer directory or HF ID (API default: Qwen/Qwen3-4B)")
    parser.add_argument("--api-base-url", help="OpenAI-compatible base URL (default: OPENAI_BASE_URL or SDK default)")
    parser.add_argument("--api-max-concurrency", type=int, default=3)
    parser.add_argument("--api-token-limit-field", choices=("max_completion_tokens", "max_tokens"),
                        default="max_completion_tokens", help="Token limit parameter accepted by the API teacher")
    parser.add_argument("--max-problems-per-source", type=int, help="Optional cap for a small run")
    parser.add_argument("--solver-batch-size", type=int, default=16, help="Number of problems per solver batch")
    parser.add_argument("--verifier-batch-size", type=int,
                        help="Problems per verifier batch (default: 1 for OpenAI, 8 for vLLM)")
    parser.add_argument("--solver-max-tokens", type=int, default=4096)
    parser.add_argument("--verifier-max-tokens", type=int, default=4096, help="Maximum tokens per verifier turn")
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-sft-tokens", type=int, default=21600)
    parser.add_argument("--validation-fraction", type=float, default=0.05, help="Fraction of retained problems")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.verifier_batch_size is None:
        args.verifier_batch_size = 1 if args.verifier_backend == "openai" else 8
    for name in ("max_problems_per_source", "solver_batch_size", "verifier_batch_size", "solver_max_tokens",
                 "verifier_max_tokens", "model_max_length", "max_sft_tokens", "tensor_parallel_size",
                 "api_max_concurrency", "solver_api_max_concurrency"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if not 0 < args.validation_fraction < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("validation fraction must be in (0, 1) and GPU memory utilization in (0, 1]")
    if args.stage != "verify":
        if args.solver_backend == "openai":
            if not args.solver_model:
                parser.error("--solver-model is required for the OpenAI solver backend")
            for variable in ("SOLVER_OPENAI_API_KEY", "SOLVER_OPENAI_BASE_URL"):
                if not os.environ.get(variable, "").strip():
                    parser.error(f"Set {variable} to use the OpenAI solver backend")
        elif args.solver_max_tokens >= args.model_max_length:
            parser.error("solver context length must leave room for prompts")
    if args.stage != "candidates":
        if args.verifier_backend == "openai":
            if not args.verifier_model:
                parser.error("--verifier-model is required for the OpenAI backend")
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                parser.error("Set OPENAI_API_KEY to use the OpenAI backend")
        elif args.max_sft_tokens + args.verifier_max_tokens + 32 > args.model_max_length:
            parser.error("verifier context length must leave room for prompts and turn generation")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transport_loggers = {"httpx", "httpx2", "httpcore"}

    def keep_log_record(record):
        return record.levelno >= logging.WARNING or record.name.split(".", 1)[0] not in transport_loggers

    for logger_name in transport_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        # Filter at the source too, in case a dependency installs its own handler.
        logger.addFilter(keep_log_record)
    for handler in logging.getLogger().handlers:
        handler.addFilter(keep_log_record)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "candidates":
        generate_candidates(args)
        return
    if args.stage == "all":
        # Exit the solver process before starting the verifier stage.
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--stage", "candidates"], check=True)
    generate_trajectories(args)


if __name__ == "__main__":
    main()
