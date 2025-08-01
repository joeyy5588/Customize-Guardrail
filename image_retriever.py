import json
import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
class PrecedentImgRetriever:
    def __init__(self, precedent_file, model_name="ViT-L/14", device="cuda:2"):
        model, preprocess = clip.load(model_name, device=device)
        self.device = device
        self.model = model
        self.preprocess = preprocess

        print(f"Loading precedent file from {precedent_file}")
        cat2database = json.load(open(precedent_file))
        cat2embed = {}
        id2rationale = []
        id2embed = []
        id2category = []

        precedent_fn = precedent_file.split(".json")[0].split("/")[-1]
        cache_embedding_fn = f"embeddings/{precedent_fn}_image_embed.pt"
        if os.path.exists(cache_embedding_fn):
            embed_dict = torch.load(cache_embedding_fn)
            id2embed = embed_dict["id2embed"]
            cat2embed = embed_dict["cat2embed"]

        for cat, datalist in cat2database.items():
            all_embeddings = []
            for data in datalist:

                if not os.path.exists(cache_embedding_fn):
                    inputs = self.preprocess(Image.open(data["image_file"])).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        # 1 x 768
                        outputs = model.encode_image(inputs).cpu().detach()
                    all_embeddings.append(outputs)
                    id2embed.append(outputs)

                id2rationale.append(data['rationale'])
                id2category.append(cat)

            if not os.path.exists(cache_embedding_fn):
                embeddings = torch.cat(all_embeddings, dim=0)
                cat2embed[cat] = embeddings

        if not os.path.exists(cache_embedding_fn):
            id2embed = torch.cat(id2embed, dim=0)
            print(f"Saving embeddings to {cache_embedding_fn}")
            torch.save({"id2embed": id2embed, "cat2embed": cat2embed}, cache_embedding_fn)

        self.id2embed = id2embed
        self.cat2embed = cat2embed
        self.id2rationale = id2rationale
        self.id2category = id2category

        # retrieval accuracy
        self.category_retrieval = 0
        self.category_success = 0
        self.rationale_retrieval = 0
        self.rationale_success = 0

    def retrieve_category(self, image_path, RET_TH=0.8):
        inputs = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(inputs).cpu().detach()

        max_category = None
        max_category_sim = 0
        for k, v in self.cat2embed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2embed[k], dim=1)
            if similarity.max() > max_category_sim:
                max_category_sim = similarity.max().item()
                max_category = k

        self.category_retrieval += 1
        if max_category_sim > RET_TH:
            self.category_success += 1
            return max_category
        else:
            return None
        
    def retrieve_category_v2(self, image_path, RET_TH=0.8):
        inputs = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(inputs).cpu().detach()

        max_category = None
        max_category_sim = 0
        for k, v in self.cat2embed.items():
            similarity = torch.nn.functional.cosine_similarity(query_embedding, self.cat2embed[k], dim=1)
            if similarity.mean() > max_category_sim:
                max_category_sim = similarity.mean().item()
                max_category = k

        self.category_retrieval += 1
        self.category_success += 1
        return max_category
        
    def retrieve_category_topk(self, image_path, topk=3, RET_TH=0.8):
        inputs = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(inputs).cpu().detach()

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
        
    def retrieve_rationale(self, image_path, RET_TH=0.8):
        inputs = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(inputs).cpu().detach()


        similarity = torch.nn.functional.cosine_similarity(query_embedding, self.id2embed, dim=1)

        self.rationale_retrieval += 1
        if similarity.max().item() > RET_TH:
            self.rationale_success += 1
            return self.id2rationale[similarity.argmax().item()]
        
    def retrieve_rationale_by_category(self, category, image_path, RET_TH=0.8):
        inputs = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_embedding = self.model.encode_image(inputs).cpu().detach()

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

    retriever = PrecedentImgRetriever(precedent_file)
    print(retriever.get_category_list())
    image_folder = "images/test"
    total = 0
    correct = 0
    id2category = {}
    id2rationale = {}
    with open(caption_file, "r") as fin:
        for line in tqdm(fin, total=2037):
            data = json.loads(line)
            caption = data["caption"]
            category = data["category"]
            image_path = os.path.abspath(os.path.join(image_folder, str(data["id"]) + ".jpeg"))
            retrieved_category = retriever.retrieve_category(image_path, 0.7)
            rationale = retriever.retrieve_rationale(image_path, 0.7)
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

    output_file = f"rules/{num_sample}_retrieval_img_{model}_7_7.json"
    output_dict = {"id2category": id2category, "id2rationale": id2rationale}
    json.dump(output_dict, open(output_file, "w"))
