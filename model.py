from tensorflow.keras.layers import LSTM , Bidirectional , Embedding, Dense ,Dropout
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import TextVectorization

def create_model(MAX_WORDS):
    model = Sequential([
        Embedding(MAX_WORDS +1 , 32),
        Bidirectional(LSTM(32, activation='tanh')),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(6, activation='sigmoid')
    ])


