import torch
import os


class Dictionary(object):
    def __init__(self):
        # 这里是创建了两个字典, 按照键值对的顺序, 第一个是word-idx, 第二个是idx-word
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            # 如果这个字典里面没有这个单词, 那么就创建一个 
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        # Add words to the dictionary
        # with open就可以不用结束阅读文本的时候手动关闭了
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                # 原来文本中每一行都是一句话
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        # 使用PyTorch库创建一个长整型（64位整数）张量
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        # 确保 ids 张量的长度是 batch_size 的整数倍, 确保批次大小一致
        return ids.view(batch_size, -1)