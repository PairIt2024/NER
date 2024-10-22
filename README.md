# (DISREGARDED) NER Pipeline with Word2Vec
This `README.md` is for a disregarded version of a Named Entity Recognition (NER) pipeline using Word2Vec and SpaCy. This version attempted to integrate Word2Vec embeddings into a SpaCy-based NER model but was eventually abandoned in favor of more efficient solutions.

# Overview
This project was an early attempt to create an NER model that leveraged Word2Vec embeddings for word representation. The goal was to extract specific entities from academic-related text, such as Professors, Classes, Sections, and Times. Although this version aimed to enhance entity recognition through dense vector embeddings, it was ultimately disregarded due to complexity and performance trade-offs.

# Features (Disregarded)
Custom NER Pipeline: A SpaCy-based pipeline that was configured to extract domain-specific entities (e.g., professors, class names, section numbers).
Word2Vec Integration: Word2Vec embeddings were utilized to represent words in vector space, with the goal of improving the model’s ability to capture semantic relationships.
Trainable NER Model: This version allowed for training custom NER models on specific datasets related to academic information.
Custom Entity Extraction: The pipeline focused on extracting specific academic entities, including Professor, Class, Section, and Time.
Setup Instructions (Disregarded)
Prerequisites
Python 3.7+: The project was developed using Python 3.7 or higher.
SpaCy: The SpaCy library was used for natural language processing and NER tasks.
Gensim: Gensim was used to train and load Word2Vec models for embedding word vectors.
OpenAI (optional): For optional integration of OpenAI embeddings, which could be used for text vectorization.
MongoDB (optional): MongoDB was used for storing the extracted data (e.g., class schedules, professor names) and related embeddings.
Installation
To set up the project, you would have installed the necessary dependencies such as SpaCy, Gensim, and pymongo for MongoDB integration. Additionally, a SpaCy language model (e.g., en_core_web_sm) would have been downloaded for text processing.

# The installation process would have involved:

Installing Python dependencies with pip.
Downloading the SpaCy language model for basic text tokenization and entity recognition.
Usage (Disregarded)
Training the NER Model
The NER pipeline allowed users to train a custom model based on annotated data that specified the locations and labels for entities like Professor, Class, and Section. The training data would be in a JSON format, containing text and annotated entities.

The pipeline would convert the annotated data into a format suitable for SpaCy and use Word2Vec embeddings to enhance the model’s performance by adding dense vector representations of words.

# Evaluating the NER Model
The model was evaluated by testing its accuracy in extracting entities from new, unseen text. The goal was to measure how effectively the model could identify and classify academic entities such as professor names, class identifiers, and class times.

# Word2Vec Embedding (Disregarded)
Word2Vec embeddings were a key feature of this approach. Word2Vec, trained using Gensim, was used to capture semantic similarities between words. The embeddings were then integrated into the NER pipeline to enrich word representations, allowing the model to better understand contextual relationships between words like professor and class.

# Model Training (Disregarded)
The model was trained using custom academic datasets. The training process involved:

Preparing annotated text data.
Converting the data into SpaCy’s Example format.
Training the NER model with Word2Vec embeddings applied to the text.
Despite promising results, this approach was eventually disregarded due to performance trade-offs and the complexity of managing Word2Vec integration within the NER pipeline.

# Contributing
This version of the NER pipeline is no longer actively maintained. Contributions are not being accepted for this version.

# License
This project is licensed under the MIT License.

