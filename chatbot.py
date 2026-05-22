from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "hello",
    "fees",
    "course",
    "certificate",
    "placement"
]

answers = [
    "Hello! How can I help you?",
    "Course fee is ₹10,000.",
    "We offer Python and Web Development courses.",
    "Yes, certificate is provided.",
    "Yes, placement support is available."
]

while True:
    user = input("You: ").lower()

    if user == "bye":
        print("Bot: Goodbye!")
        break

    text = questions + [user]

    cv = CountVectorizer()
    matrix = cv.fit_transform(text)

    similarity = cosine_similarity(matrix[-1], matrix[:-1])

    index = similarity.argmax()

    print("Bot:", answers[index])