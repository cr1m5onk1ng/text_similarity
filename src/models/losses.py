import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.distance import CosineSimilarity
from src.configurations import config as config
from src.configurations.config import Configuration, ModelParameters
from src.dataset.dataset import *
from src.training.learner import ModelOutput, ClassifierOutput, SimilarityOutput

class Loss(nn.Module):
    def __init__(self, params: Configuration):
        super(Loss, self).__init__()
        self.params = params

    def forward(self, hidden_state: torch.Tensor, features: EmbeddingsFeatures) -> ModelOutput:
        raise NotImplementedError()


class SoftmaxLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = nn.Linear(self.params.model_parameters.hidden_size*3, self.params.model_parameters.num_classes)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, hidden_state, features):
        labels = features.labels 
        logits = self.classifier(hidden_state)
        loss = self.loss_function(
            logits, 
            labels.view(-1)
        )
        return ClassifierOutput(
            loss = loss,
            predictions = logits
        )
        

class SimilarityLoss(Loss):
    def __init__(self, *args, margin=0.5, **kwargs):
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
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings
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
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings,
        ) 


class CosineSimilarityLoss(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_function = nn.MSELoss()

    def forward(self, embeddings, features):
        scores = self.similarity(embeddings[0], embeddings[1])
        if isinstance(features, dict):
            labels = features["labels"]
        else:
            labels = features.labels
        loss = self.loss_function(scores, labels.view(-1))
        return ClassifierOutput(
            loss = loss,
            predictions = embeddings
        )


class SimpleDistillationLoss(Loss):
    """
    Distillation loss based on a simple MSE loss
    between the teacher and student embeddings
    """
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss = nn.MSELoss()

    def forward(self, student_embeddings, features):
        teacher_embeddings = features.generate_labels(self.teacher_model)
        loss = self.loss(student_embeddings, teacher_embeddings)
        return ClassifierOutput(
            loss=loss, 
            predictions=torch.stack([student_embeddings, teacher_embeddings], dim=0)
        )



class FastDistillationLoss(Loss):
    """
    Distillation loss based on Fastformers approach
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.loss_function = nn.MSELoss()
        self.alpha_ce = self.params.alpha_ce
        self.alpha_clm = self.params.alpha_clm
        self.alpha_mse = self.params.alpha_mse
        self.alpha_cos = self.params.alpha_cos

        self.loss = None #the actual loss to optimize
        self.last_loss_ce = 0
        self.last_loss_clm = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        self.last_log = 0

        self.ce_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = torch.nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = torch.nn.CosineEmbeddingLoss(reduction="mean")

    def forward(self, student_logits, teacher_logits, features):
        mask = features.attention_mask.unsqueeze(-1).expand_as(student_logits)
        s_logits_slct = torch.masked_select(student_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, student_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(teacher_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, student_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = self.ce_loss_fct(
            F.log_softmax(s_logits_slct / self.params.temperature, dim=-1),
            F.softmax(t_logits_slct / self.params.temperature, dim=-1)
        ) * (self.params.temperature) ** 2

        loss = self.alpha_ce * loss_ce

        #WTF
        if self.alpha_clm > 0.0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = features.labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm

        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse

        #PROBLEMA: IL MODELLO DEVE DARE IN OUTPUT GLI HIDDEN STATES COME
        #AVVIENE NELLA LIBRERIA TRANSFORMERS
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = features.attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new_ones(s_hidden_states_slct.size(0)) # (bs * seq_length,)
            loss_cos = self.cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
            loss += self.alpha_cos * loss_cos
        return ModelOutput(loss)


    




