import os
import json
import pickle
import argparse
import re
from transformers import AutoTokenizer 
from vllm import LLM, SamplingParams

from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer, _normalize


# VLLM Setup for cluster environments
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

class LLMJudge:
    def __init__(
        self,
        base_model_name_or_path: str,
        num_gpus: int = 1,
        context_size: int = 32000,
    ):
        self.model_name = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            local_files_only=True,
        )
        
        # VLLM Initialization
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=int(num_gpus),
            trust_remote_code=True,
            max_model_len=context_size,
            dtype='bfloat16',
            gpu_memory_utilization=0.9, 
            enforce_eager=True,
        )
        
        # Define sampling params once
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            skip_special_tokens=True 
        )

    def format_prompt(self, question, gold_answer, model_answer) -> str:
        chat = [
            {
                "role": "system",
                "content": (
                    "You are an impartial judge for Question Answering. "
                    "Compare the Candidate Answer to the Reference Answer.\n"
                    "Your output must be strict JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Evaluate the semantic equivalence of the following answers.\n\n"
                    f"Question: {question}\n"
                    f"Reference Answer: {gold_answer}\n"
                    f"Candidate Answer: {model_answer}\n\n"
                    "Criteria:\n"
                    "1. Ignore minor phrasing differences.\n"
                    "2. If the Candidate Answer contains the Reference Answer's core meaning, it is correct.\n"
                    "3. If the Candidate Answer contradicts the Reference, it is incorrect.\n\n"
                    "Respond with this exact JSON format:\n"
                    "{\"correct\": <bool>, \"explanation\": \"<short string>\"}"
                ),
            }
        ]
        return self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    def predict_batch(self, questions, gold_answers, model_answers):
        prompts = [
            self.format_prompt(q, g, m) 
            for q, g, m in zip(questions, gold_answers, model_answers)
        ]

        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input pickle")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-14B", help="Judge model ID")
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join("nanochat_evaluations", "eval_results_scored.pkl"),
        help="Path to write the scored pickle.",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)
    output_filename = args.output_file
    print(f"We will save to: {output_filename}")
    # 1. Load Data
    print(f"Loading results from {args.input_file}...")
    with open(args.input_file, "rb") as f:
        data = pickle.load(f)

    tokenizer = SimpleTokenizer()
    judge = LLMJudge(args.judge_model)

    for condition_name, condition_data in data.items():
        print(f"\nScoring condition: {condition_name}")

        predictions = condition_data["results"]

        ######################### 
        # Exact Match Accuracy
        #########################
        exact_match_results = [
            has_answers(pred["prediction"], pred["answers"], tokenizer=tokenizer, regex=False)
            for pred in predictions
        ]

        ######################### 
        # LLM-Judge Accuracy
        #########################
        questions = [pred["question"] for pred in predictions]
        gold_answers = [pred["answers"] for pred in predictions]
        model_answers = [pred["prediction"] for pred in predictions]

        judge_responses = judge.predict_batch(
            questions=questions,
            gold_answers=gold_answers,
            model_answers=model_answers
        )

        for pred, em_score, judge_score in zip(predictions, exact_match_results, judge_responses):
            pred["exact_match_score"] = em_score
            pred["llm_judge_score"] = judge_score

        print(f"Saved scored results to {output_filename}")

    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved scored results to {output_filename}")


    
