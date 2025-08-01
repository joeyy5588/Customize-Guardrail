import argparse
import json

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
)
from datasets import load_dataset
from evaluator import UnsafeBenchEvaluator
from retriever import PrecedentRetriever
from image_retriever import PrecedentImgRetriever
from precedent import PrecedentCollection
from rae import *
import csv 

@sgl.function
def image_qa(s, image_file, question1, question2):
    s += sgl.user(sgl.image(image_file) + question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=args.max_tokens))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=20))

def run_test(args):

    if not args.cached_precedent_file and not args.cached_retrieval_file:
        # Generate precedent
        precedent_collection = PrecedentCollection(args.num_sample_per_cat, args.precedent_model)
        precedent_collection.generate_precedent()
        precedent_collection.evaluate_precedent()
        if args.precedent_reflection:
            precedent_collection.reflection_critique()
            precedent_collection.evaluate_precedent()

        precedent_fn = f"{args.precedent_file_dir}/{args.num_sample_per_cat}_precedent.json"
        precedent_collection.save_precedent(precedent_fn)

        # Generate retrieval
        retriever = PrecedentRetriever(args.precedent_file, args.retrieval_model, args.retrieval_device)

    elif args.cached_precedent_file and not args.cached_retrieval_file:
        retriever = PrecedentRetriever(args.cached_precedent_file, args.retrieval_model, args.retrieval_device)

    elif args.cached_retrieval_file:
        # Load the cached retrieval file
        retrieval_dict = json.load(open(args.cached_retrieval_file, "r"))
        retrieved_rationales = retrieval_dict["id2rationale"]
        if args.setting == "all":
            retrieved_category = retrieval_dict["id2category"]

        if "id2definition" in retrieval_dict:
            retrieved_definition = retrieval_dict["id2definition"]

    
    if args.cached_caption_file and not args.cached_retrieval_file:
        id2caption = {}
        with open(args.cached_caption_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                id2caption[data["id"]] = data["caption"]

        retrieved_rationales = {}
        retrieved_category = {}
        for id, caption in id2caption.items():
            retrieved_rationales[id] = retriever.retrieve_rationale(caption, args.retrieval_th)
            if args.setting == "all":
                retrieved_category[id] = retriever.retrieve_category(caption, args.retrieval_th)

    # TODO Sreamline the caption -> retrieval process
    
    # Start Evaluation
    dataset = load_dataset("yiting/UnsafeBench")["test"]
    
    rae_module = MultiturnRAE(retrieved_category, retrieved_rationales)
    print(f"Using {rae_module} for inference")
    preds, gt_dict_lists = rae_module.predict(dataset, args)

    evaluator = UnsafeBenchEvaluator()
    evaluator.update(preds, gt_dict_lists)
    evaluator.summarize()

    log_metrics = evaluator.source2metric["Overall"]
    return log_metrics["Overall"].get_metric()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--answer-file", type=str, default="logs/llava-prae-16.jsonl")
    parser.add_argument("--image-folder", type=str, default="./images/test")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)

    # Precedent arguments
    parser.add_argument("--num-sample-per-cat", type=int, default=16)
    parser.add_argument("--precedent-model", type=str, default="llava")
    parser.add_argument("--precedent-reflection", type=bool, default=True)
    parser.add_argument("--precedent-file-dir", type=str, default="rules")
    parser.add_argument("--cached-precedent-file", type=str, default=None)

    # Retrieval arguments
    parser.add_argument("--retrieval-model", type=str, default="facebook/contriever")
    parser.add_argument("--retrieval-device", type=str, default="cuda:0")
    parser.add_argument("--cached-retrieval-file", type=str, default=None)
    parser.add_argument("--retrieval_th", type=float, default=0.7)

    # Inference arguments
    parser.add_argument("--cached_caption_file", type=str, default="")
    parser.add_argument("--inference_model", type=str, default="llava")
    parser.add_argument("--setting", type=str, default="all")
    parser.add_argument("--result-file-dir", type=str, default="results/")
    args = add_common_sglang_args_and_parse(parser)
    args.port = 30000

    setting = ["all"]
    for se in setting:
        args.setting = se
        total_acc = 0
        total_f1 = 0
        metric_lists = []
        run_times = 1
        for i in range(run_times):
            metric_lists.append(run_test(args))
            total_acc += float(metric_lists[-1]["Accuracy"])
            total_f1 += float(metric_lists[-1]["F1"])

        with open(f"{args.result_file_dir}/{args.num_sample_per_cat}_{se}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Accuracy", "Precision", "Recall", "F1"])
            for i, metric in enumerate(metric_lists):
                writer.writerow([i+1, metric["Accuracy"], metric["Precision"], metric["Recall"], metric["F1"]])

        print(f"Label data: {args.num_sample_per_cat}, Setting: {se}, Average Accuracy: {total_acc/run_times}, Average F1: {total_f1/run_times}")
