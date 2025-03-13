import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from emailenv import EmailEnv
from qlearningagent import QLearningAgent
from icecream import ic

# * load and preprocess email data
emails = pd.read_csv("../data/emails.csv")
products = pd.read_csv("../data/demoitemdata.csv")

# * extract features from emails
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails["content"])
y = emails["labels"]

# * encode labels
# ? original labels
labels = ["items", "noItems"]

# ? init LabelEncoder
label_encoder = LabelEncoder()

# ? fit the encoder on the labels
label_encoder.fit(labels)

# ? transform the labels to numerical values 
encoded_labels = label_encoder.transform(labels)

# ? print the mapping
for original_label, encoded_label in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    ic(f"{original_label} -> {encoded_label}")

# ? convert numerical labels back to original labels 
original_labels = label_encoder.inverse_transform(encoded_labels)
y = label_encoder.fit_transform(y)

# * split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# * train the model
env = EmailEnv(X_train, y_train)
agent = QLearningAgent(env.action_space, env.observation_space)

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_actions(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state


# * deploy model
def respond_to_email(email_content):
    email_vector = vectorizer.transform([email_content]).toarray().flatten()
    ic(email_vector)
    action = agent.choose_actions(email_vector)
    ic(action)
    if action == 1:
        # * generate response using product data
        response = "here are the product details..."
        return response
    else:
        return "no quote detected"


new_email = "Hi there, please can you quote"
response = respond_to_email(new_email)
ic(response)
