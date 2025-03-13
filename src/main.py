import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from emailenv import EmailEnv
from qlearningagent import QLearningAgent
from icecream import ic

# * load and preprocess email data
emails = pd.read_csv()
products = pd.read_csv("/data/demoitemdata.csv")

# * extract features from emails
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails["content"])
y = emails["label"]

# * encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# * split dat into training and testing sets
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
    action = agent.choose_actoin(email_vector)
    if action == 1:
        # * generate response using product data
        response = "here are the product details..."
        return response
    else:
        return "no quote detected"


new_email = ""
response = respond_to_email(new_email)
ic(response)
