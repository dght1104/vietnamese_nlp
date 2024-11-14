
import pandas as pd
data_path = 'models/shopping_rating.csv'

# Đọc dữ liệu từ file CSV
data = pd.read_csv(data_path)

# In thử 4 dòng đầu tiên của dữ liệu
print(data.head(4))

# Tiếp tục các bước xử lý và huấn luyện...
X = data['comment']  # Cột chứa câu văn
y = data['sentiment']  # Cột chứa nhãn cảm xúc