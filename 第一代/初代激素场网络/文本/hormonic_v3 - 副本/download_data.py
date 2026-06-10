import urllib.request
import os

os.makedirs('./data', exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
print('Downloading...')
urllib.request.urlretrieve(url, './data/tinyshakespeare.txt')
print('Done!')

with open('./data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f'Size: {len(text)} chars')
print(f'Preview: {text[:200]}...')
