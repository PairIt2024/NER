import spacy
import json
from spacy.training import Example, offsets_to_biluo_tags

def load_test_data(json_file_path):
    # Load the annotated test data from a JSON file
    with open(json_file_path, 'r') as f:
        test_data = json.load(f)
    train_data = []
    for entry in test_data:
        text = entry['text']
        #Format is "start", end, "label"
        for ent in entry['entities']:
            entities = [(ent['start'], ent['end'], ent['label'])]
            annotations = {"entities": entities}
        train_data.append((text, annotations))
    return train_data


def evaluate_ner_model(nlp, test_data):
    scorer = spacy.scorer.Scorer()  # Initialize the SpaCy Scorer to evaluate the model
    test_data = load_test_data("../data/annotated_test_data.json")
    print(test_data)

    examples = []
    # Loop through the test data and create Example objects for scoring
    for text, annotations in test_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
        # Create a Doc object from the text
        doc = nlp.make_doc(text)

       
    scores = nlp.evaluate(examples)

    # Print the evaluation scores
    print(f"Precision: {scores['ents_p']:.3f}")
    print(f"Recall: {scores['ents_r']:.3f}")
    print(f"F1 Score: {scores['ents_f']:.3f}")

   

if __name__ == "__main__":
    # Load the trained NER model
    nlp = spacy.load("../models/ner_model")

    # Load the annotated test data
    test_data = load_test_data("../data/annotated_test_data.json")

    # Evaluate the model on the test data
    evaluate_ner_model(nlp, test_data)
