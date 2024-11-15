import tensorflow as tf 
from Save import load_tokenizer 
import gradio as gr

from train import df

model = tf.keras.models.load_model('toximodel.h5')
vectorizer = load_tokenizer('tokenizer.pkl')

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