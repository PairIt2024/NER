import spacy
import json
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score
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

def evaluate_model(nlp, examples):
    true_labels = []
    pred_labels = []

    for text, annotations in examples:
        doc = nlp(text)
        true_entities = [ent[2] for ent in annotations['entities']]
        pred_entities = [ent.label_ for ent in doc.ents]

        true_labels.extend(true_entities)
        pred_labels.extend(pred_entities)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    return acc, f1

def train_ner_model(k=5, epochs=15):
    try:
        # Load the training data
        TRAIN_DATA = load_train_data('../data/annotated_data.json')
        print("Training data has been completed")
        
        # Split the data into training and validation sets (80/20)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold = 1
        for train_index, val_index in kf.split(TRAIN_DATA):
            print(f"\nTraining on fold {fold}...")

            train_data = [TRAIN_DATA[i] for i in train_index]
            val_data = [TRAIN_DATA[i] for i in val_index]

            # Initialize a blank SpaCy model for NER
            print("Training NER model")
            nlp = spacy.blank("en")

            # Load the Word2Vec model
            print("Loading word2vec model")
            word2vec_model = Word2Vec.load("../data/word2vec.model")

            # Add NER to the pipeline
            print("Adding NER to the pipeline")
            ner = nlp.add_pipe("ner")

            # Add labels for the NER model (SECTION and PROFESSOR)
            print("Adding labels")
            ner.add_label("SECTION")
            ner.add_label("PROFESSOR")

            # Resize vectors to match Word2Vec
            nlp.vocab.vectors.resize((len(nlp.vocab), word2vec_model.vector_size))

            # Add Word2Vec vectors to SpaCy vocab
            for word in word2vec_model.wv.index_to_key:
                if word in nlp.vocab:
                    nlp.vocab[word].vector = word2vec_model.wv[word]

            # Convert training data to SpaCy's Example objects
            train_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in train_data]
            val_examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in val_data]

            # Initialize the optimizer
            optimizer = nlp.begin_training()

            for epoch in range(epochs):
                random.shuffle(train_examples)
                losses = {}

                # Update model using minibatches
                batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    nlp.update(batch, sgd=optimizer, drop=0.35, losses=losses)

                # Evaluate on the validation set
                val_acc, val_f1 = evaluate_model(nlp, val_examples)
                
                # Print metrics after each epoch
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {losses['ner']:.4f} - Validation Accuracy: {val_acc:.4f} - Validation F1 Score: {val_f1:.4f}")
            
            fold += 1
        
        # Save the model after training
        nlp.to_disk("../models/ner_model")
        return nlp
    
    except Exception as e:
        print("Error training NER model:", e)
        return None

if __name__ == "__main__":
    # Train the model using K-Fold cross-validation and 15 epochs
    train_ner_model(k=5, epochs=15)