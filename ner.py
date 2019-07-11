# -*- coding:utf-8 -*-
import os
import re
import sys
from nltk.tag import StanfordNERTagger


# modelfile = '/Users/freesiasouth/stanfordnlp_resources/stanford-chinese-corenlp-2018-10-05-models/' \
#            'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz'
modelfile = 'model/boson-ner-tal0.01-model.ser.gz'
jar = '/Users/freesiasouth/stanfordnlp_resources/stanford-ner-2018-10-16/stanford-ner.jar'
path_data = 'data/text/'
path_ner = 'data/text_with_ner/'


def split_text(text):
    words = []
    w = [i for i in text]
    temp = ''
    for i in range(len(w)):
        if re.match('[\u4e00-\u9fa5]', w[i]) or re.match("[ \.\!\/_,$%^*\(\)+\"\']+|[+——！，。？、~@#￥%……&*（）：;；=><\t]+", w[i]):
            if temp is not '':
                words.append(temp)
                temp = ''
            words.append(w[i])
        else:
            temp += w[i]
    return words


def do_ner(path_data, path_ner, modelfile, jar):
    tagger = StanfordNERTagger(modelfile, jar)
    flist = os.listdir(path_data)
    length = len(flist)
    for i, file in enumerate(flist):
        ners = []
        reader = open(path_data+file, 'r')
        words = split_text(reader.readline())
        reader.close()
        tags = tagger.tag(words)
        curr = 0
        for w in words:
            if w is tags[curr][0]:
                ners.append(tags[curr])
                curr += 1
            else:
                ners.append((w, 'O'))
        right = True
        if len(words) != len(ners):
            right = False
        for w, n in zip(words, ners):
            if w != n[0]:
                right = False
                break
        if right:
            writer = open(path_ner + file, 'w')
            for n in ners:
                writer.write(n[0] + '\t:\t' + n[1] + '\n')
            writer.close()
            print(i, '/', length)
        else:
            print(file, sys.stderr)


def write_instance(writer, words, tag):
    for word in words:
        word = word.strip()
        if len(word) > 0:
            writer.write(word + '\t' + tag + '\n')


def get_train_set():
    isTag = False
    text = ''
    writer = open('/Users/freesiasouth/Documents/ner-train-set.txt', 'w')
    reader = open('/Users/freesiasouth/Downloads/BosonNLP_NER_6C/BosonNLP_NER_6C.txt', 'r')
    for line in reader:
        line = line.strip()
        for i in range(len(line)):
            if line[i] == '{' and line[i+1] == '{' and not isTag:
                if text.startswith('}}'):
                    text = text[2:]
                words = split_text(text)
                write_instance(writer, words, 'O')
                text = ''
                isTag = True
            if i == len(line)-1 and line[i] == '}' and line[i-1] == '}' and isTag:
                text = text[2:]
                tag = text.split(':')[0]
                words = split_text(text.split(':')[1])
                write_instance(writer, words, tag)
                text = ''
                isTag = False
            if i != len(line)-1 and line[i] == '}' and line[i+1] == '}' and isTag:
                text = text[2:]
                tag = text.split(':')[0]
                words = split_text(text.split(':')[1])
                write_instance(writer, words, tag)
                text = ''
                isTag = False
            text += line[i]
    writer.close()
    reader.close()


def get_ner_tags():
    tags = set()
    with open('/Users/freesiasouth/Documents/ner-train-set.txt', 'r') as reader:
        for line in reader:
            tags.add(line.split('\t')[-1].strip())
    tags = list(tags)
    tags.sort()
    print(tags)


if __name__ == '__main__':
    # do_ner(path_data, path_ner, modelfile, jar)
    # get_train_set()
    get_ner_tags()

