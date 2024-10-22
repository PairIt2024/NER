from gensim.models import Word2Vec

#Train word2vec model on the given tokens
def train_embedding_model(all_tokens):
    #Model is Skip-gram
    model = Word2Vec(sentences=all_tokens,vector_size=100, window=5, min_count=1, workers=4, sg=1)
    return model

#Embed all the tokens with the given trained model
def embed_tokens(tokens, model):
    return [model.wv[token] for token in tokens if token in model.wv]