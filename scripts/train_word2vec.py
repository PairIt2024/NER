from db_config import connectToDB
from embedding import embed_tokens, train_embedding_model
from fetch_data import fetch_data

def train_word2vec_model():
    try:
        #Connect to DB
        collection = connectToDB()
        if collection is None:
            print("Error connecting to DB")
            return
        
        #Fetch data and fill lists
        data_list, all_tokens = fetch_data(collection)

        if data_list is None or all_tokens is None:
            print("Error fetching data")
            return

        #Train word2vec model
        model = train_embedding_model(all_tokens)

        embedded_data = []
        #Embed all the tokens
        for data in data_list:
            #Embed each section and professor with the given model that was trained
            #Expecting a vector from embed_tokens
            embedded_section = embed_tokens(data['tokens_section'], model)
            embedded_professor = embed_tokens(data['tokens_professor'], model)

            print("Embedded section: ", embedded_section)
            print("Embedded professor: ", embedded_professor)

            embedded_data.append({'embedded_section': embedded_section, 'embedded_professor': embedded_professor})
        print("Embedded data: ", embedded_data)
        model.save("../data/word2vec.model")

    except Exception as e:
        print("Error in main: ", e)
    
if __name__ == "__main__":
    train_word2vec_model()