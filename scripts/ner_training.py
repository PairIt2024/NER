import spacy
import json
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
from gensim.models import Word2Vec

def load_train_data(json_file_path):
    #Open the json file and load the data
    print("Loading training data")
    with open(json_file_path) as f:
        data = json.load(f)
    train_data = []
    #Format the data for training
    print("Formatting data")
    for entry in data:
        print("Entry: ", entry)
        text = entry['text']
        # Format is "start", end, "label"
        for ent in entry["entities"]:
            print("Entity: ", ent)
            entities = [(ent['start'], ent['end'], ent['label'])]
            annotations = {"entities": entities}
        train_data.append((text, annotations))
    return train_data

def train_ner_model():
    try:
        #Train the NER model utilizing the word2vec model trained with all section and classes
        print("Training NER model")
        nlp = spacy.blank("en")

        #Load the word2vec model
        print("Loading word2vec model")
        word2vec_model = Word2Vec.load("../data/word2vec.model")

        #Add NER to the pipeline
        print("Adding NER to the pipeline")
        ner = nlp.add_pipe("ner")

        #Labels for the NER model (section and professor)
        print("Adding labels")
        ner.add_label("SECTION")
        ner.add_label("PROFESSOR")

        #match length of word2vec
        nlp.vocab.vectors.resize((len(nlp.vocab), word2vec_model.vector_size))

        #Add vector to the vocab
        for word in word2vec_model.wv.index_to_key:
            if word in nlp.vocab:
                nlp.vocab[word].vector = word2vec_model.wv[word]

        #Prepare the training data (need to add)
        TRAIN_DATA = load_train_data('../data/annotated_data.json')
        print("Training data has been completed")

        #Convert to Example objects
        TRAIN_DATA_EXAMPLES = []
        for text, annotation in TRAIN_DATA:   
            print(text, annotation) 
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotation)
            TRAIN_DATA_EXAMPLES.append(example)

        optimizer = nlp.begin_training()

        for itn in range(15):
            random.shuffle(TRAIN_DATA_EXAMPLES)
            losses = {}
            batches = minibatch(TRAIN_DATA_EXAMPLES, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, sgd = optimizer, drop = 0.35, losses = losses)
            print("Losses", losses)
            print("Iteration: ", itn)
        
        nlp.to_disk("../models/ner_model")
        #Save the model
        return nlp
    except Exception as e:
        print("Error training NER model: ", e)
        return None
    
if __name__ == "__main__":
    train_ner_model()
    



