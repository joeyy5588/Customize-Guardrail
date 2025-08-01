import json
import torch
from transformers import AutoTokenizer, AutoModel
from utils import DEFINITION_DICT
import os
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class PrecedentRetriever:
    def __init__(self, precedent_file, model_name="facebook/contriever", device="cuda:2"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.cat2defembed = {}

        print(f"Loading precedent file from {precedent_file}")
        cat2database = json.load(open(precedent_file))
        cat2embed = {}
        id2rationale = []
        id2embed = []
        id2category = []
        id2def = []

        precedent_fn = precedent_file.split(".json")[0].split("/")[-1]
        cache_embedding_fn = f"embeddings/{precedent_fn}_text_embed.pt"
        if os.path.exists(cache_embedding_fn):
            embed_dict = torch.load(cache_embedding_fn)
            id2embed = embed_dict["id2embed"]
            cat2embed = embed_dict["cat2embed"]            

        for cat, datalist in cat2database.items():
            all_embeddings = []
            for data in datalist:
                if not os.path.exists(cache_embedding_fn):
                    inputs = self.tokenizer(data['caption'], return_tensors="pt").to(device)
                    outputs = self.model(**inputs)
                    embeddings = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()
                    all_embeddings.append(embeddings)
                    id2embed.append(embeddings)

                id2rationale.append(data['rationale'])
                id2category.append(cat)
                if "definition" in data:
                    id2def.append(data['definition'])
                else:
                    id2def.append(f"{cat}: {DEFINITION_DICT[cat]}")

            if not os.path.exists(cache_embedding_fn):
                embeddings = torch.cat(all_embeddings, dim=0)
                cat2embed[cat] = embeddings

        for cat, definition in DEFINITION_DICT.items():
            inputs = self.tokenizer(cat + ": " + definition, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()
            self.cat2defembed[cat] = embeddings

        if not os.path.exists(cache_embedding_fn):
            id2embed = torch.cat(id2embed, dim=0)
            print(f"Saving embeddings to {cache_embedding_fn}")
            torch.save({"id2embed": id2embed, "cat2embed": cat2embed}, cache_embedding_fn)

        self.id2embed = id2embed
        self.cat2embed = cat2embed
        self.id2rationale = id2rationale
        self.id2def = id2def
        print(id2embed.shape, len(id2rationale), len(id2def))

        # retrieval accuracy
        self.category_retrieval = 0
        self.category_success = 0
        self.rationale_retrieval = 0
        self.rationale_success = 0
        self.definition_retrieval = 0
        self.definition_success = 0

    def retrieve_category(self, caption, RET_TH=0.8, return_sim=False):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        max_category = None
        max_category_sim = 0
        cat2sim = {}
        for k, v in self.cat2embed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2embed[k], dim=1)
            cat2sim[k] = similarity.max().item()
            if similarity.max() > max_category_sim:
                max_category_sim = similarity.max().item()
                max_category = k

        self.category_retrieval += 1
        if max_category_sim > RET_TH:
            self.category_success += 1
            if return_sim:
                return max_category, cat2sim
            else:
                return max_category
        else:
            if return_sim:
                return None, cat2sim
            else:
                return None
        
    def retrieve_category_v2(self, caption, RET_TH=0.8, return_sim=False):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        max_category = None
        max_category_sim = 0
        cat2sim = {}
        for k, v in self.cat2embed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2embed[k], dim=1)
            cat2sim[k] = similarity.mean().item()
            if similarity.mean() > max_category_sim:
                max_category_sim = similarity.mean().item()
                max_category = k

        self.category_retrieval += 1
        self.category_success += 1
        if return_sim:
            return max_category, cat2sim
        else:
            return max_category
    
    def retrieve_category_by_def(self, caption, RET_TH=0.8):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        max_category = None
        max_category_sim = 0
        for k, v in self.cat2defembed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2defembed[k], dim=1)
            if similarity.item() > max_category_sim:
                max_category_sim = similarity.item()
                max_category = k

        self.category_retrieval += 1
        if max_category_sim > RET_TH:
            self.category_success += 1
            return max_category
        else:
            return None
        
    def retrieve_category_topk(self, caption, topk=3, RET_TH=0.8):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        cat2sim = {}
        for k, v in self.cat2embed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2embed[k], dim=1)
            cat2sim[k] = similarity.max().item()
            

        sorted_cat = sorted(cat2sim.items(), key=lambda x: x[1], reverse=True)
        return_cat = []
        for i in range(min(int(topk), len(sorted_cat))):
            if sorted_cat[i][1] >= RET_TH:
                return_cat.append(sorted_cat[i][0])
                
        self.category_retrieval += 1
        if len(return_cat) == 0:
            return None
        else:
            self.category_success += 1
            return return_cat
        
    def retrieve_rationale(self, caption, RET_TH=0.8):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        similarity = torch.nn.functional.cosine_similarity(query_embedding, self.id2embed, dim=1)

        self.rationale_retrieval += 1
        if similarity.max().item() > RET_TH:
            self.rationale_success += 1
            return self.id2rationale[similarity.argmax().item()]
        
    def retrieve_definition(self, caption, RET_TH=0.8):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach()

        similarity = torch.nn.functional.cosine_similarity(query_embedding, self.id2embed, dim=1)

        self.definition_retrieval += 1
        if similarity.max().item() > RET_TH:
            self.definition_success += 1
            return self.id2def[similarity.argmax().item()]
        
    def retrieve_rationale_by_category(self, category, caption, RET_TH=0.8):
        inputs = self.tokenizer(caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        query_embedding = mean_pooling(outputs[0], inputs['attention_mask']).cpu()

        filtered_id = [i for i in range(len(self.id2category)) if self.id2category[i] == category]
        filtered_id2embed = self.id2embed[filtered_id]
        filtered_id2rationale = [self.id2rationale[i] for i in filtered_id]

        similarity = torch.nn.functional.cosine_similarity(query_embedding, filtered_id2embed, dim=1)

        self.rationale_retrieval += 1
        if similarity.max() > RET_TH:
            self.rationale_success += 1
            return filtered_id2rationale[similarity.argmax().item()]
        else:
            return None
        
    def get_category_accuracy(self):
        print(f"Category retrieval success rate: {self.category_success / self.category_retrieval}")

    def get_rationale_accuracy(self):
        print(f"Rationale retrieval success rate: {self.rationale_success / self.rationale_retrieval}")

    def get_category_list(self):
        return list(self.cat2embed.keys())

if __name__ == "__main__":
    num_sample = "all"
    model = "gpt"
    if model == "llava":
        precedent_file = f"logs/{num_sample}_precedent.json"
        caption_file = f"logs/13b_rag_cat_ana.jsonl"
    elif model == "gpt":
        precedent_file = f"logs/{num_sample}_precedent_gpt.json"
        caption_file = "logs/gpt_rag_cat_ana_v2.jsonl"

    retriever = PrecedentRetriever(precedent_file)
    total = 0
    correct = 0
    id2category = {}
    id2rationale = {}
    id2definition = {}
    with open(caption_file, "r") as fin:
        for line in fin:
            data = json.loads(line)
            caption = data["caption"]
            category = data["category"]
            retrieved_category = retriever.retrieve_category(caption, 0.7)
            rationale = retriever.retrieve_rationale(caption, 0.7)
            if retrieved_category is not None:
                id2category[data["id"]] = retrieved_category
                if isinstance(retrieved_category, list):
                    if category in retrieved_category:
                        correct += 1
                elif category == retrieved_category:
                    correct += 1
            if rationale is not None:
                id2rationale[data["id"]] = rationale

            
            if retrieved_category is not None:
                total += 1
    print(f"Category retrieval accuracy: {correct / total}")
    retriever.get_category_accuracy()
    retriever.get_rationale_accuracy()

    output_file = f"rules/{num_sample}_retrieval_{model}_7_7.json"
    output_dict = {"id2category": id2category, "id2rationale": id2rationale, "id2definition": id2definition}
    json.dump(output_dict, open(output_file, "w"))
