from src.utils import utils
from src.configurations import classifier_config as config
from src.configurations import embeddings_config as embeddings_config
from src.dataset.dataset import *
from src.modules.contextual_embedder import ContextualEmbedder
from src.evaluation.evaluators import RetrievalEvaluator
from src.models.modeling import SiameseSentenceEmbedder, MBERTClassifier
from src.utils.metrics import RetrievalAccuracyMeter, cos_sim
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = {
    "mbert_avg": "mbert_sense_features_avg",
    "mbert_cls": "mbert_sense_cls_features",
    "baseline": "mbert_no_sense_cls",
    "siamese_contrastive_cls": "training_siamese_paws_sense_online_contrastive_cls_pooling_5epochs"
}

MODEL_NAME = MODEL_NAMES["baseline"]
PRETRAINED_PATH = f"../training/trained_models/{MODEL_NAME}"
    
metrics = {"validation": [RetrievalAccuracyMeter]}

model = MBERTClassifier(
    strategy="cls",
    train_model = False,
    use_sense_embeddings=False
)

model.load_pretrained(PRETRAINED_PATH)

model.to(config.DEVICE)

model.eval()


language_pairs = ['eng-jpn', 'eng-deu', 'eng-fra', 'eng-spa', 'eng-kor']
dev_paths = [f'../data/tatoeba/parallel-sentences/Tatoeba-{pair}-dev.tsv.gz' for pair in language_pairs]

langs = {
    "jp": 0,
    "de": 1,
    "fr": 2,
    "es": 3,
    "ko": 4
}

lang = langs["jp"]

valid_dataset = ParallelDataset.build_dataset([dev_paths[lang]])

valid_data_loader = SmartParaphraseDataloader.build_batches(valid_dataset, 16, mode="tatoeba")

metrics = {"validation": [RetrievalAccuracyMeter]}

print(f"Lang: {language_pairs[lang]}")

evaluator = RetrievalEvaluator(
    model = model,
    data_loader=valid_data_loader,
    device=config.DEVICE,
    metrics=metrics,
    fp16=True,
    verbose=True,
    return_predictions = False

)

evaluator.evaluate()

"""
for d in valid_data_loader:
    with torch.no_grad():
        encoded = model.encode(d["sentence_1_features"], d["sentence_2_features"])
    encoded_1 = encoded[0]
    encoded_2 = encoded[1]
    src_encoded.append(encoded_1)
    tgt_encoded.append(encoded_2)

src_encoded = torch.cat(tuple(src_encoded), dim=0)
tgt_encoded = torch.cat(tuple(tgt_encoded), dim=0)


cos_sims = pytorch_cos_sim(src_encoded, tgt_encoded).detach().cpu().numpy()
#print(f"Cos sims: {cos_sims}")
correct_src2trg = 0
correct_trg2src = 0
for i in range(len(cos_sims)):
    max_idx = np.argmax(cos_sims[i])
    
    if i == max_idx:
        correct_src2trg += 1
    
    print("i:", i, "j:", max_idx, "INCORRECT" if i != max_idx else "CORRECT")
    print("Src:", source_sentences[i])
    print("Trg:", target_sentences[max_idx])
    print("Argmax score:", cos_sims[i][max_idx], "vs. correct score:", cos_sims[i][i])

    results = zip(range(len(cos_sims[i])), cos_sims[i])
    results = sorted(results, key=lambda x: x[1], reverse=True)
    for idx, score in results[0:5]:
        print("\t", idx, "(Score: %.4f)" % (score), self.target_sentences[idx])
    
cos_sims = cos_sims.T
for i in range(len(cos_sims)):
    max_idx = np.argmax(cos_sims[i])
    if i == max_idx:
        correct_trg2src += 1
acc_src2trg = correct_src2trg / len(cos_sims)
acc_trg2src = correct_trg2src / len(cos_sims)
print("Accuracy src2trg: {:.2f}".format(acc_src2trg*100))
print("Accuracy trg2src: {:.2f}".format(acc_trg2src*100))


"""
