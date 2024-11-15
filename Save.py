import pickle 

def save_tokenizer(tokenizer , filename):
    with open(filename  , 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Tokenizer is Saved")

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {filename}")
    return tokenizer

def save_model(model , filename):
    if filename.endswith == '.h5':
        model.save(filename)
    else:
        filename+'.h5'

        model.save(filename)
    

