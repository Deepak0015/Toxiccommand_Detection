import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.layers import TextVectorization
from model import create_model
from Save import save_model
from Save import save_tokenizer

df = pd.read_csv('train.csv')
X = df['comment_text']
y = df.drop(columns = ['comment_text' , 'id'], axis = 1)

y = y.values 
MAX_WORDS = 200000  #number of words in vocab
vectorizer = TextVectorization(max_tokens=MAX_WORDS , #number of words in the vocab
    output_sequence_length = 1800 , #it the length of the outsequense  
    output_mode = 'int'#the word number is integear
    )

vectorizer.adapt(X.values)


model  = create_model(MAX_WORDS)
model.compile(loss='BinaryCrossentropy', optimizer='Adam',metrics=["accuracy"])
model.summary()
vectorized_text = vectorizer(X)
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text,y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)

train_size = int(len(dataset) * 0.6)  # 60% for training
val_size = int(len(dataset) * 0.2)    # 20% for validation
test_size = len(dataset) - train_size - val_size  # Remaining for testing

train = dataset.take(train_size)

remaining_dataset = dataset.skip(train_size)
val = remaining_dataset.take(val_size)

remaining_dataset = remaining_dataset.skip(val_size)
test = remaining_dataset.take(test_size)
history = model.fit(train, epochs=1, validation_data=val)

save_tokenizer(vectorizer, filename='tokenizer.pkl')
save_model(model=model , filename='toxicmodel.h5')
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)
