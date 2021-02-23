from konlpy.tag import Komoran

def word_ngram(bow, num_gram):
    text = tuple(bow)
    ngrams = [text[x:x+ num_gram] for x in range(0, len(text))]
    return tuple(ngrams)

def similarity(doc1, doc2):
    cnt = 0
    for token in doc1:
        if token in doc2:
            cnt = cnt +1
    return cnt/len(doc1)


sentence1 = "6월에 뉴턴은 선생님의 제안으로 트리니티에 입학했다."
sentence2 = "6월에 뉴턴은 선생님의 제안으로 대학교에 입학했다."
sentence3 = "나는 맛있는 밥을 뉴턴 선생님과 함께 먹었다. "

komoran = Komoran()
bow1 = komoran.nouns(sentence1)
bow2 = komoran.nouns(sentence2)
bow3 = komoran.nouns(sentence3)

doc1 = word_ngram(bow1, 2)
doc2 = word_ngram(bow2, 2)
doc3 = word_ngram(bow3, 2)

print(doc1)
print(doc2)


r1 = similarity(doc1, doc2)
r2 = similarity(doc3, doc1)

print(r1)
print(r2)
