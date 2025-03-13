import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from emailenv import EmailEnv
from qlearningagent import QLearningAgent
from icecream import ic
from scipy.sparse import hstack
import numpy as np

# * load and preprocess email data
emails = pd.read_csv("../data/emails.csv")
products = pd.read_csv("../data/demoitemdata.csv")

# * extract features from emails
vectorizer_content = CountVectorizer()
vectorizer_subject = CountVectorizer()
vectorizer_requestedItems = CountVectorizer()
X_content = vectorizer_content.fit_transform(emails["content"])
X_subject = vectorizer_subject.fit_transform(emails["subject"])
X_requestedItems = vectorizer_requestedItems.fit_transform(emails["requestedItems"])
X_isAllowedDomain = emails["isAllowedDomain"].values.reshape(-1, 1)

# * combine features into a single feature matrix
X = hstack([X_content, X_subject, X_requestedItems, X_isAllowedDomain])

y = emails["labels"]

# * encode labels
# ? original labels
labels = ["noItems", "items"]

# ? init LabelEncoder
label_encoder = LabelEncoder()

# ? fit the encoder on the labels
label_encoder.fit(labels)

# ? transform the labels to numerical values
encoded_labels = label_encoder.transform(labels)

# ? print the mapping
for original_label, encoded_label in zip(
    label_encoder.classes_, range(len(label_encoder.classes_))
):
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

for episode in range(20000):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_actions(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state


# * deploy model
def respond_to_email(
    email_content, email_subject, email_requestedItems, email_isAllowedDomain
):
    email_vector_content = vectorizer_content.transform([email_content]).toarray().flatten()
    email_vector_subject = vectorizer_subject.transform([email_subject]).toarray().flatten()
    email_vector_requestedItems = (
        vectorizer_requestedItems.transform([email_requestedItems]).toarray().flatten()
    )
    email_vector_isAllowedDomain = (
        np.array([email_isAllowedDomain]).reshape(-1, 1).flatten()
    )

    email_vector = np.concatenate(
        [
            email_vector_content,
            email_vector_subject,
            email_vector_requestedItems,
            email_vector_isAllowedDomain,
        ]
    )
    ic(email_vector)
    action = agent.choose_actions(email_vector)
    ic(action)
    if action == 0:
        # * generate response using product data
        response = "here are the product details..."
        return response
    else:
        return "no quote detected"

# * process each email and return a response
responses = [] 
for index, row in emails.iterrows(): 
    response = respond_to_email(row["content"], row["subject"], row["requestedItems"], row["isAllowedDomain"])
    responses.append(response)

for response in responses: 
    ic(response)