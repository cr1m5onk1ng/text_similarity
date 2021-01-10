
#from src.configurations import embeddings_config as config
import src.utils.utils as utils

if __name__ == '__main__':


    embed_map = utils.load_pretrained_embeddings("../embeddings/ares_embedding/ares_bert_base_multilingual.txt", 768*2, reduction=False)
    utils.save_file(embed_map, "../embeddings", 'ares_embed_map')
    mappa = utils.load_file("../embeddings/ares_embed_map")

    print(len(list(mappa.keys())), len(list(mappa.values())[0]), list(mappa.keys())[:10])