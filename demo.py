# demo.py

from vietnamese_nlp.sentiment_lib import predict_sentiment

# Nhập câu văn để dự đoán cảm xúc
new_text = "Hôm nay tôi không vui"

# Dự đoán cảm xúc cho câu văn
result = predict_sentiment(new_text)

# In kết quả
print(f"Dự đoán cảm xúc: {result}")
