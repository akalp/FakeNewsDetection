import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


def read():
    # df = pd.read_csv("./data/fake_news_train.csv")
    df = pd.read_csv("./output/4max.csv")
    # df["title"] = df["title"].fillna("-NO TITLE-")
    # df["author"] = df["author"].fillna("-NO AUTHOR-")
    # df["text"] = df["text"].fillna("-NO TEXT-")
    # df["all_text"] = df["title"] + " " + df["text"]
    # df = df.loc[:, ("id", "all_text", "label")]
    return df


def train_validation_split(df, train_ratio):
    train_size = round(df.shape[0] * train_ratio)
    _train = df.iloc[:train_size, :].groupby('label')
    validation = df.iloc[train_size:, :]
    return _train.get_group(0), _train.get_group(1), validation


def tfidf_dict(data, type, stopword_bool):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                 stop_words=ENGLISH_STOP_WORDS) if stopword_bool else TfidfVectorizer(
        ngram_range=(1, 1))
    if type is "bigram":
        vectorizer = TfidfVectorizer(ngram_range=(2, 2),
                                     stop_words=ENGLISH_STOP_WORDS) if stopword_bool else TfidfVectorizer(
            ngram_range=(2, 2))
    vectors = vectorizer.fit_transform(data)

    counts = vectors.sum(axis=0).A1
    return dict(zip(vectorizer.get_feature_names(), counts))


def train2(_real, _fake, stopword_bool):
    real_uni_dict = tfidf_dict(_real.all_text, "unigram", stopword_bool)
    real_bi_dict = tfidf_dict(_real.all_text, "bigram", stopword_bool)

    fake_uni_dict = tfidf_dict(_fake.all_text, "unigram", stopword_bool)
    fake_bi_dict = tfidf_dict(_fake.all_text, "bigram", stopword_bool)

    return real_uni_dict, real_bi_dict, fake_uni_dict, fake_bi_dict


def count_dict(data, _type, stopword_bool):
    vectorizer = CountVectorizer(ngram_range=(1, 1),
                                 stop_words=ENGLISH_STOP_WORDS) if stopword_bool else CountVectorizer(
        ngram_range=(1, 1))
    if _type is "bigram":
        vectorizer = CountVectorizer(ngram_range=(2, 2),
                                     stop_words=ENGLISH_STOP_WORDS) if stopword_bool else CountVectorizer(
            ngram_range=(2, 2))
    vectors = vectorizer.fit_transform(data)

    counts = vectors.sum(axis=0).A1
    return dict(zip(vectorizer.get_feature_names(), counts))


def train(_real, _fake, stopword_bool):
    real_uni_dict = count_dict(_real.all_text, "unigram", stopword_bool)
    real_bi_dict = count_dict(_real.all_text, "bigram", stopword_bool)

    fake_uni_dict = count_dict(_fake.all_text, "unigram", stopword_bool)
    fake_bi_dict = count_dict(_fake.all_text, "bigram", stopword_bool)

    return real_uni_dict, real_bi_dict, fake_uni_dict, fake_bi_dict


def part_1(_real, _fake):
    r_title_uni_dict = count_dict(_real.title, "unigram", False)
    f_title_uni_dict = count_dict(_fake.title, "unigram", False)

    print("Total word count in title of real news:", sum(r_title_uni_dict.values()))
    print("Unique word count in title of real news:", len(r_title_uni_dict.keys()))
    print("Total word count in title of fake news:", sum(f_title_uni_dict.values()))
    print("Unique word count in title of fake news:", len(f_title_uni_dict.keys()))

    diff_r_f = dict(
        sorted({key: r_title_uni_dict[key] for key in list(set(r_title_uni_dict) - set(f_title_uni_dict))}.items(),
               key=lambda k: k[1], reverse=True))
    diff_f_r = dict(
        sorted({key: f_title_uni_dict[key] for key in list(set(f_title_uni_dict) - set(r_title_uni_dict))}.items(),
               key=lambda k: k[1], reverse=True))
    for i in list(diff_r_f.keys())[:3]:
        print("{}: frequency on real: {}, fake: 0".format(i, diff_r_f[i]))
    for i in list(diff_f_r.keys())[:3]:
        print("{}: frequency on fake: {}, real: 0".format(i, diff_f_r[i]))


def classify(_real_freq, _fake_freq, _val, real_news_count, fake_news_count, type):
    # KAGGLE (return the list)
    # pred_list = []

    r_total_word_count = sum(_real_freq.values())
    f_total_word_count = sum(_fake_freq.values())
    unique_word_count = len(set(list(_real_freq.keys())) - set(list(_fake_freq.keys())))
    real_prob = real_news_count / (real_news_count + fake_news_count)
    fake_prob = fake_news_count / (real_news_count + fake_news_count)

    correct_pred_count = 0

    n = 1 if type is "unigram" else 2
    for index, row in _val.iterrows():
        r_pred, f_pred = np.log(real_prob), np.log(fake_prob)

        tokens = CountVectorizer(ngram_range=(n,n))
        tokens.fit_transform([row["all_text"]])
        for token in tokens.get_feature_names():
            r_count = _real_freq[token] if token in _real_freq else 0
            f_count = _fake_freq[token] if token in _fake_freq else 0

            r_pred += np.log((r_count + 1) / (r_total_word_count + unique_word_count))
            f_pred += np.log((f_count + 1) / (f_total_word_count + unique_word_count))

        pred = 0 if r_pred > f_pred else 1

        # KAGGLE
        # pred_list.append(pred)

        if pred == row['label']:
            correct_pred_count += 1

    return correct_pred_count


def cal_accuracy(correct, total):
    return 100 * (correct / total)


def main():
    data_df = read()
    real, fake, validation = train_validation_split(df=data_df, train_ratio=0.9)

#%% KAGGLE
    # real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)
    #
    # validation = pd.read_csv("./data/fake_news_test_file.csv")
    # validation = validation.drop(validation[validation["title"] == validation["text"]].index)
    # validation["title"] = validation["title"].fillna("-NO TITLE-")
    # validation["author"] = validation["author"].fillna("-NO AUTHOR-")
    # validation["text"] = validation["text"].fillna("-NO TEXT-")
    # validation["all_text"] = validation["title"] + " " + validation["text"]
    # validation = validation.loc[:, ("id", "all_text")]
    #
    # l = classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram")
    # df = pd.DataFrame(l, index=validation.id, columns=["label"])
    # df.to_csv("./output/kaggle.csv")

#%% test

    # real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)
    #
    # uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram"),
    #                    validation.shape[0])
    #
    # print(uni)

#%% homework
    # part_1(real, fake)
    #
    real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram"),
                       validation.shape[0])

    bi = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "bigram"),
                      validation.shape[0])

    print("CountVectorizer w/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train2(real, fake, False)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram"),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "bigram"),
                      validation.shape[0])

    print("TfidfVectorizer w/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train(real, fake, True)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram"),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "bigram"),
                      validation.shape[0])

    print("CountVectorizer wo/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train2(real, fake, True)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram"),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "bigram"),
                      validation.shape[0])

    print("TfidfVectorizer wo/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))
