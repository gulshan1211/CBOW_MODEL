import numpy as np
import tensorflow as tf
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define global variables
stop_words = set(stopwords.words('english'))
pronouns = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'}

# Function to preprocess a question
def preprocess_question(question):
    # Remove punctuation and special characters
    question = re.sub(r'[^\w\s]', '', question)
    # Convert to lowercase
    question = question.lower()
    # Tokenize the question
    question_words = question.split()
    # Part-of-speech tagging and lemmatization
    pos_tags = nltk.pos_tag(question_words)
    lemmas = []
    for word, tag in pos_tags:
        wn_tag = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        lemmas.append(lemma)
    # Remove stop words and pronouns
    filtered_words = [word for word in lemmas if word not in stop_words and word not in pronouns]
    return ' '.join(filtered_words)

# Function to convert POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'  # default to noun if tag is not found

# Example sentences data (replace with your actual data)
sentences = {'sentences': [
    "The cat sat on the mat.",
    "The dog barked at the cat.",
    # Add more sentences as needed
]}

# Preprocess questions
preprocessed_questions = [preprocess_question(q) for q in sentences['sentences']]

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_questions)
sequences = tokenizer.texts_to_sequences(preprocessed_questions)
vocab_size = len(tokenizer.word_index)

# Generate context-target pairs for Skip-gram model
window_size = 1
contexts = []
targets = []

for sequence in sequences:
    for i in range(window_size, len(sequence) - window_size):
        context = sequence[i - window_size:i] + sequence[i + 1:i + window_size + 1]
        target = sequence[i]
        contexts.append(context)
        targets.append(target)

# Convert context and target to one-hot encoding
X = np.zeros((len(contexts), vocab_size), dtype=int)
Y = np.zeros((len(targets), vocab_size), dtype=int)

for idx, context in enumerate(contexts):
    for i in context:
        X[idx, i - 1] = 1

for idx, target in enumerate(targets):
    Y[idx, target - 1] = 1

# Define the Skip-gram model using Keras Sequential API
model = Sequential([
    Dense(2, input_shape=(vocab_size,), activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=20, batch_size=4)

# Retrieve the weight matrix and transpose it
a = np.transpose(model.layers[1].get_weights()[0])

# Function to normalize each row of a matrix
def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms

# Normalize the weight matrix 'a' along axis 0 (rows)
normalized_a = normalize_rows(a)

# Print normalized matrix shape and example row
print("Normalized matrix shape:", normalized_a.shape)
print("Example normalized row:", normalized_a[0])

# Function to calculate cosine similarity and write results to a file
def calculate_cosine_similarity_and_write(embeddings, tokenizer, output_file):
    num_words = len(embeddings)
    similarities = np.zeros((num_words, num_words))

    # Check if the output file exists, create it if not
    if not os.path.exists(output_file):
        with open(output_file, 'w'):  # Create the file
            pass

    with open(output_file, 'a') as file:
        for i in range(num_words):
            for j in range(num_words):
                if i != j:
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities[i, j] = similarity
                    word1 = list(tokenizer.word_index.keys())[i]
                    word2 = list(tokenizer.word_index.keys())[j]
                    file.write(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}\n")

    return similarities

# Calculate cosine similarity between all pairs of words
output_file = 'cosine_similarities.txt'
embeddings = [normalized_a[idx - 1] for idx in sorted(tokenizer.word_index.values())]
similarities = calculate_cosine_similarity_and_write(embeddings, tokenizer, output_file)

# Print the similarity matrix
print("\nSimilarity matrix:")
print(similarities)
