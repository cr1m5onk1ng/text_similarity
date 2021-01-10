import torch
from torch import nn
from torch.nn import functional as F
from src.configurations import config as config
from src.configurations.config import ModelParameters
from src.dataset.dataset import *
from src.training.learner import ModelOutput, ClassifierOutput, SimilarityOutput

class Loss(nn.Module):
    def __init__(self, params: ModelParameters):
        super(Loss, self).__init__()
        self.params = params

    def forward(self, hidden_state: torch.Tensor, features: EmbeddingsFeatures) -> ModelOutput:
        raise NotImplementedError()


class SoftmaxLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.params.hidden_size, self.params.num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, hidden_state, features):
        labels = features.labels 
        logits = self.classifier(hidden_state)
        loss = self.loss_function(
            logits.view(-1, self.params.num_classes), 
            labels.view(-1)
        )
        return ClassifierOutput(
            loss = loss,
            predictions = logits
        )
        

class SimilarityLoss(Loss):
    def __init__(margin=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def forward(self, embeddings, features):
        raise NotImplementedError()


class ContrastiveSimilarityLoss(SimilarityLoss):
    """Ranking loss based on the measure of cosine similarity """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, features):
        assert embeddings.shape[0] == 2
        distances = 1 - F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
        loss = 0.5 * (features.labels.float() * distances.pow(2) + (1 - features.labels).float() * F.relu(self.margin - distances).pow(2))
        return SimilarityOutput(
            loss = loss,
            embeddings = torch.stack([embeddings[0], embeddings[1]], dim=0)
        ) 


class OnlineContrastiveSimilarityLoss(SimilarityLoss):
    """Online contrastive loss as defined in SBERT """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, embeddings, features):
        assert embeddings.shape[0] == 2
        distance_matrix = 1-F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)
        negs = distance_matrix[features.labels == 0]
        poss = distance_matrix[features.labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]
        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return SimilarityOutput(
            loss = loss,
            embeddings = torch.stack([embeddings[0], embeddings[1]], dim=0)
        ) 


class CosineSimilarityLoss(SimilarityLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_function = nn.MSELoss()

    def forward(self, embeddings, features):
        scores = self.similarity(embeddings[0], embeddings[1])
        loss = self.loss_function(scores, features.labels)
        return SimilarityOutput(
            loss = loss,
            scores = scores
        )







