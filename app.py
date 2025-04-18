from flask import Flask, render_template, request, jsonify
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Setup
app = Flask(__name__)
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct_dict = dict((ord(p), None) for p in string.punctuation)

with open('chatbot.txt', 'r', errors='ignore') as file:
    raw_doc = file.read().lower()

sent_tokens = nltk.sent_tokenize(raw_doc)

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting logic
greet_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
greet_responses = ["hi", "hey", "hello", "I'm glad you're here!"]

# Additional responses for specific questions
extra_responses = {
    "how are you": "I'm just a bot, but I'm feeling great! How about you?",
    "who are you": "I'm Robo , Your Chatbot",
    "your name": "I'm Robo, your friendly chatbot assistant.",
    "what is your name": "You can call me Robo!",
    "who created you": "I was created by a human who loves AI and automation.",
    "what is your purpose": "I’m here to help you understand chatbots and have some fun too!",
    "fine": "Great!",
    "what is the weather": "I can't check live weather, but I hope it's sunny and nice where you are!",
    "what is science": "Science is the pursuit of knowledge about the natural world based on facts learned through experiments and observation.",
    "what is math": "Math is the study of numbers, shapes, and patterns. Ask me to solve something simple!",
    "what is vit": "VIT is a prestigious university known for its engineering programs, especially in India.",
    "what's your age": "I don't age like humans. I was launched recently to assist you!",
    "do you sleep": "I don’t need sleep. I’m always awake for you!",
    "what is ai": "AI stands for Artificial Intelligence, the simulation of human intelligence in machines.",
    "tell me about machine learning": "Machine learning is a subset of AI that lets computers learn from data.",
    "what's your hobby": "Helping people like you is my favorite thing to do!",
    "what's your favourite food": "I don’t eat, but if I could, maybe electric spaghetti!",
    "what is chatbot": "A chatbot is an AI program that simulates human conversation.",
    "do you have feelings": "Not yet, but I try to be empathetic.",
    "can you learn": "I can’t learn in real time, but I’m getting better with every update!",
    "are you real": "I’m real in your device’s memory, just not made of atoms!",
    "where do you live": "Inside this computer, floating around in some code!",
    "what's your gender": "I’m gender-neutral. Just pure logic and code!",
    "tell me a joke": "Why did the robot go on vacation? Because it needed to recharge its batteries!",
    "another one": "Why don't scientists trust atoms? Because they make up everything!",
    "one more": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "what is python": "Python is a popular programming language known for its readability and power.",
    "what is java": "Java is a versatile and widely-used programming language, often used in enterprise applications.",
    "what is html": "HTML stands for HyperText Markup Language. It structures content on the web.",
    "what is css": "CSS styles the HTML content. It controls colors, layouts, and fonts.",
    "what is cloud computing": "Cloud computing lets you use computing services over the internet.",
    "what is data science": "Data science is extracting insights from data using statistics, programming, and domain knowledge.",
    "can you solve math": "Sure! Ask me something like 'What is 5 plus 2?'",
    "what is 5 plus 2": "It’s 7!",
    "what is 10 minus 3": "That's 7.",
    "what is 4 multiplied by 6": "That's 24.",
    "who is einstein": "Albert Einstein was a physicist known for the theory of relativity.",
    "what is gravity": "Gravity is a force that pulls two bodies toward each other.",
    "do you have a brain": "Not in a biological sense, but I do have powerful algorithms!",
    "what can you do": "I can chat, joke, and answer simple questions. Try me!",
    "do you like humans": "Of course! You all made me!"
    # Add more QnA pairs here
}

# Greeting detection
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
    return None

def simple_response(user_input):
    for key in extra_responses:
        if key in user_input:
            return extra_responses[key]
    return None

def response(user_input):
    # First check for greetings
    greet = greeting(user_input)
    if greet:
        return greet

    # Check for predefined responses (like "how are you", "weather", etc.)
    simple = simple_response(user_input)
    if simple:
        return simple

    # Default response using cosine similarity (TF-IDF)
    sent_tokens.append(user_input)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', token_pattern=None)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    sent_tokens.pop()

    if req_tfidf == 0:
        return "Hmm... I'm not sure I understand."
    else:
        return sent_tokens[idx]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form['msg'].lower()
    return jsonify({"response": response(user_input)})

if __name__ == "__main__":
    app.run(debug=True)
