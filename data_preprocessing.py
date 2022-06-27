# Importing Required Libraries
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

# try:
#     nltk.download('omw-1.4')
# except Exception as e:
#     print('Package is already up-to-date!')

# Loading Data
data_file = open('intents.json').read()
intents  = json.loads(data_file)

# Create Empty list to Append Words, Docs and Classes...
words = []
classes = []
documents = []
# ignore punctuation from words
ignore_words = ['?',',','!','.','-']

# Data Preprocessing
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize all words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Associate Token Group with Tag
        documents.append((w, intent['tag'])) 
        # Save Unique Tags as Classes: Add the Tags to Classes List
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
# Convert Words to their Root which is Not in ignore_words  
clean_words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# Remove Duplicate Words: ie. Unique Words in Above Words list
clean_words = list(set(clean_words))
# Remove Duplicate Classes
classes = list(set(classes))

# Save Clean Words to Pickle File
pickle.dump(clean_words, open('words.pkl', 'wb'))
# Save Classes to Pickle File
pickle.dump(classes, open('classes.pkl', 'wb'))
# Save Associate Token Group with Tag to Pickle File
pickle.dump(documents, open('documents.pkl', 'wb'))