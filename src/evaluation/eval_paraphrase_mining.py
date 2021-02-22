from src.dataset.dataset import ParaphraseProcessor, SmartParaphraseDataloader
from src.models.sentence_encoder import SiameseSentenceEmbedder
from src.configurations import config
import random
import parser
import torch
import transformers


if __name__ == "__main__":

    parser.add_argument('--fp16', type=bool, dest="mixed_precision", default=True)
    parser.add_argument('--seq_len', type=int, dest="seq_len", default=256)
    parser.add_argument('--device', type=str, dest="device", default="cuda")
    parser.add_argument('--model', type=str, dest="model", default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument('--pretrained-model-path', type=str, dest="pretrained_model_path", default="trained_models/sbert-jp-jsnli/sbert-jp-jsnli.bin")
    args = parser.parse_args()

    model_config = config.SenseModelParameters(
        model_name = args.config_name,
        hidden_size = args.hidden_size,
        num_classes = 3,
        use_pretrained_embeddings = args.use_pretrained_embeddings,
        freeze_weights = args.freeze_weights,
        context_layers = (-1,)
    )

    configuration = config.Configuration(
        model_parameters=model_config,
        model = args.model,
        save_path = args.save_path,
        sequence_max_len = args.seq_len,
        dropout_prob = args.dropout,
        lr = args.lr,
        batch_size = args.batch_size,
        epochs = args.epochs,
        device = torch.device(args.device),
        embedding_map = None,
        bnids_map = None,
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model),
        pretrained_embeddings_dim = config.DIMENSIONS_MAP[args.sense_embeddings_type],
        senses_as_features = args.senses_as_features
    )

    embedder_config = transformers.AutoConfig.from_pretrained(configuration.model)
    embedder = transformers.AutoModel.from_pretrained(configuration.model, config=embedder_config)

    model = SiameseSentenceEmbedder(
        params = configuration,
        context_embedder=embedder
    )

    model.load_pretrained(args.pretrained_model_path)
    

    num_queries = 100

    corpus_percent = .7

    document_dataset = ParaphraseProcessor.build_document_dataset_from_paws("pawspath")

    #TODO implement in dataloader
    eval_dataloader = SmartParaphraseDataloader.build_batches(document_dataset, 16, mode="document")

    positive_sentences = document_dataset.positive_examples
    negative_sentences = document_dataset.negative_examples

    all_sentences = positive_sentences + negative_sentences

    random.shuffle(all_sentences)

    all_sentences = all_sentences[:int(len(all_sentences)*corpus_percent)]

    for i in range(num_queries):
        query = random.randint(0, len(positive_sentences))
        encoded_query = model.encode_text(query.text)
        