# vietnamese_nlp/tokenization.py
from pyvi import ViTokenizer

class VietnameseTokenizer:
    @staticmethod
    def tokenize(text):
        return ViTokenizer.tokenize(text)


# import subprocess
# import json

# class VietnameseTokenizer:
#     def __init__(self):
#         # Đường dẫn đến file VnCoreNLP-1.1.1.jar
#         self.vncorenlp_path = "D:\\Y4 HK1\\VnCoreNLP\\VnCoreNLP-1.1.1.jar"

#     def tokenize(self, text):
#         # Gọi VnCoreNLP để tách từ
#         process = subprocess.Popen(
#             ["java", "-Xmx2g", "-jar", self.vncorenlp_path, "-a", "wseg"],
#             stdin=subprocess.PIPE,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )

#         # Gửi văn bản đến VnCoreNLP và nhận kết quả
#         stdout, stderr = process.communicate(text)
        
#         if stderr:
#             print("Error:", stderr)
#             return []

#         # Chuyển đổi kết quả JSON thành danh sách từ
#         annotations = json.loads(stdout)
#         tokens = [word['form'] for word in annotations[0]['tokens']]
        
#         return tokens
