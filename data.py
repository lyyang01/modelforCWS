import sys, pickle, os, random
import numpy as np
import codecs

## tags, BIEO
#定义tag2label = {"O": 0,
#             "B": 1, "I": 2, "E": 3}


def load_example(words): #词数组，得到x,y #用于训练
    y=[]
    for word in words:
        if len(word)==1: y.append(3)
        else: y.extend([0]+[1]*(len(word)-2)+[2])
    return ''.join(words),y

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    注意：data即为feed给模型的train_data
    """
    data = []
    with codecs.open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        #temp = line.split()
        if line != '\n':
        #if temp[0] != '。':
            try:
                #[char, label] = line.strip().split()
                seqs,label=load_example(line.strip().split())
            except:
                continue
            sent_= seqs
            tag_=label
            #print([char, label])
        
        data.append((sent_, tag_))
    sent_, tag_ = [], []
   

    return data
#test
#data = read_corpus('pku_training.utf8')
#read_corpus('train_my_data')


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    """word2id  [字的id号,该字词频]"""
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

#
#vocab_build('.//vocab', 'pku_training.utf8', 5)
#vocab_build('.//vocab_my', './/train_my_data', 5)


'''对句子进行编码'''
def sentence2id(sent, word2id):
   
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id

'''读取建立的词库'''
def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    #print('vocab_size:', len(word2id))
    #print(word2id)
    return word2id

#read_dictionary('.//vocab')

def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

'''对于长度不够的句子进行padding'''
def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, shuffle=False):
    """
    生成批量数据，送入模型训练
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        #label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(tag_)

    if len(seqs) != 0:
        yield seqs, labels
   
    
#test code    

#test_path = os.path.join('.', 'data_path', 'test_my_data')
#test_data = read_corpus(test_path); test_size = len(test_data)
#word2id = read_dictionary('vocab')
#print(word2id)
#batches = batch_yield(data, 3, word2id, shuffle=False)
#for step, (seqs, labels) in enumerate(batches):
#    word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
#print(3)
