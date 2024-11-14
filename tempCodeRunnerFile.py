texts = ["Xin chào, tôi là sinh viên.", "Tôi thích học lập trình.", "Xin chào, hôm nay bạn có khỏe không?"]

# Khởi tạo đối tượng TextVectorizer
vectorizer = TextVectorizer()

# Học và biến đổi văn bản thành vectơ TF-IDF
vectorized_texts = vectorizer.fit_transform(texts)
print(vectorized_texts.toarray())

# Chuyển đổi một văn bản mới thành vectơ TF-IDF
new_text_vector = vectorizer.transform("Xin chào, hôm nay bạn có khỏe không?")
print(new_text_vector.toarray())
