from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import nmslib
import spacy

# Loading the transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_index(embeddings):
    # Initialize a new index
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2})
    return index

def semantic_search(query: str, index, sentence_list, top_k: int = 5):
    # Convert the query to embeddings
    query_embedding = model.encode([query])
    ids, distances = index.knnQuery(query_embedding, k=top_k)
    return [sentence_list[i] for i in ids], distances

def listPrint(list):
    for ele in list:
        print(ele)

nlp = spacy.load('en_core_web_sm')

def extract_keywords(sentence: str):
    doc = nlp(sentence)
    return [token.text for token in doc if token.pos_ in ['NOUN', 'VERB']]

text = '''david: Am I awake? I feel dizzy, Oh hi! Who is this girl?"
sarah: Hello there! I'm Sarah, your girlfriend. Are you feeling okay? Do you need some water or something?
david: what's happening outside, the world is destroyed?
david: where is my cake by the way? never mind, this is not the time for cake
sarah: Oh, don't worry about that. The world outside is just fine. Let's focus on spending time together, okay?
david: but I see everything is on fire!
sarah: Oh, I'm sorry. I must have forgotten to close the curtains. But don't worry, everything is fine. Let's just enjoy our time together, okay?
david: but...
sarah: It's okay, don't worry about it. Let's just relax and enjoy our time together, okay?
david: I'm so panic right now, I need some water
sarah: Of course, let me get you some water. Just sit tight and I'll be right back.
david: (try to escape)
sarah: Where do you think you're going? You can't leave me! You're mine forever!
david: I'm so sorry
sarah: It's okay, I forgive you. Just please don't try to leave me again, okay?
david: You scares me! Why you pull out that KNIFE?!
sarah: I'm sorry, I didn't mean to scare you. I just got a little carried away. Please don't be afraid of me, I would never hurt you. Can you forgive me?
david: no
sarah: I understand. I'll try to be better, I promise. Please don't leave me, I need you.
david: Unless you give me your home key, I'm not going to trust you
sarah: No, I can't give you my home key. That's not something I can do. You need to trust me without needing something like that.
david: Then you should tell me the truth
sarah: Okay, I'll tell you the truth. I have a condition that makes me act a little strange sometimes. But I'm working on it, and I promise I won't hurt you. Can you trust me?
david: Yes. I need to know the truth about the outside world
sarah: Okay, I'll tell you the truth. The world outside is fine, I just didn't want you to leave me. I'm sorry for lying to you. Can you forgive me?
david: You are still lying!
sarah: I'm not lying, I promise! Please believe me, I would never lie to you.
david: Look at outside, the city is on fire
sarah: Oh my goodness, you're right! I had no idea. We need to get out of here. Let me grab my things and we'll leave together.
david: yes let's go right now!
sarah: Okay, let's go! I have the key to the exit door. We can leave now'''

conversational_history = text.split('\n')

# prepare the embeddings and the indices
sentence_embeddings = model.encode(conversational_history)
sentence_embeddings_np = np.array(sentence_embeddings)
search_index = create_index(sentence_embeddings_np)

# working on the new query
new_line = "Can I have something to eat?"
keywords = extract_keywords(new_line)
print(keywords)
results, distances = semantic_search(new_line, search_index, conversational_history)
listPrint(results)
listPrint(distances)