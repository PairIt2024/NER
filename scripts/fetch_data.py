from preprocess import preprocess_and_tokenize

#Fetch data from the DB
def fetch_data(collection):
    try:
        #Fetch all the data from the DB
        data_cursor = collection.find()

        #For Word2Vec model
        all_tokens = []

        #Tokenized list of all the data
        data_list = []

        #For all the data in db
        for data in data_cursor:
            #Preprocess and tokenize the data
            section = data.get('section','')
            professor = data.get('instructor','')

            #Preprocess and tokenize the data
            tokens_section = preprocess_and_tokenize(section)
            tokens_professor = preprocess_and_tokenize(professor)

            #Collect tokens
            all_tokens.append(tokens_section)
            all_tokens.append(tokens_professor)

            #Store data for embedding
            data_list.append({"tokens_section":tokens_section, "tokens_professor":tokens_professor})

        return data_list, all_tokens
        
    except Exception as e:
        print("Error fetching data: ", e)
        return None
    