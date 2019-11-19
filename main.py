import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS


def read():
    df = pd.read_csv("./output/17.csv")
    # df = pd.read_csv("./data/fake_news_train.csv")
    # df["title"] = df["title"].fillna("-NO TITLE-")
    # df["author"] = df["author"].fillna("-NO AUTHOR-")
    # df["text"] = df["text"].fillna("-NO TEXT-")
    # df["all_text"] = df["title"] + " " + df["text"]
    df = df.loc[:, ("id", "all_text", "label")]
    return df


def train_validation_split(df, train_ratio):
    # df = df.sample(frac=1).reset_index(drop=True) # randomize rows
    # df.to_csv("./output/{}.csv".format(i))
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
    r_uni_dict = count_dict(_real.all_text, "unigram", False)
    f_uni_dict = count_dict(_fake.all_text, "unigram", False)

    print("Total word count in real news:", sum(r_uni_dict.values()))
    print("Unique word count in real news:", len(r_uni_dict.keys()))
    print("Total word count in fake news:", sum(f_uni_dict.values()))
    print("Unique word count in fake news:", len(f_uni_dict.keys()))

    diff_r_f = dict(
        sorted({key: r_uni_dict[key] for key in list(set(r_uni_dict) - set(f_uni_dict))}.items(),
               key=lambda k: k[1], reverse=True))
    diff_f_r = dict(
        sorted({key: f_uni_dict[key] for key in list(set(f_uni_dict) - set(r_uni_dict))}.items(),
               key=lambda k: k[1], reverse=True))
    for i in list(diff_r_f.keys())[:3]:
        print("{}: frequency on real: {}, fake: 0".format(i, diff_r_f[i]))
    for i in list(diff_f_r.keys())[:3]:
        print("{}: frequency on fake: {}, real: 0".format(i, diff_f_r[i]))


def classify(_real_freq, _fake_freq, _val, real_news_count, fake_news_count, type, kaggle):
    # KAGGLE (return the list)
    pred_list = []

    r_total_word_count = sum(_real_freq.values())
    f_total_word_count = sum(_fake_freq.values())
    unique_word_count = len(set(list(_real_freq.keys())) - set(list(_fake_freq.keys())))
    real_prob = real_news_count / (real_news_count + fake_news_count)
    fake_prob = fake_news_count / (real_news_count + fake_news_count)

    correct_pred_count = 0

    for index, row in _val.iterrows():
        r_pred, f_pred = np.log(real_prob), np.log(fake_prob)

        tokens = count_dict([row["all_text"]], type, False)
        for token in tokens.keys():
            r_count = _real_freq[token] if token in _real_freq else 0
            f_count = _fake_freq[token] if token in _fake_freq else 0

            r_pred += (np.log((r_count + 1) / (r_total_word_count + unique_word_count)))*tokens[token]
            f_pred += (np.log((f_count + 1) / (f_total_word_count + unique_word_count)))*tokens[token]

        pred = 0 if r_pred > f_pred else 1

        # KAGGLE
        pred_list.append(pred)
        if not kaggle:
            if pred == row['label']:
                correct_pred_count += 1

    if not kaggle:
        return correct_pred_count
    return pred_list


def cal_accuracy(correct, total):
    return 100 * (correct / total)


def test_randomize_data(data_df):
    from matplotlib import pyplot as plt

    X = np.arange(20)
    y = []
    for i in X:
        real, fake, validation = train_validation_split(df=data_df, train_ratio=0.9, i=str(i))

        real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)

        uni = cal_accuracy(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", False),
                           validation.shape[0])
        y.append(uni)
        print("{}\t:{}".format(i, uni))

    fig, ax = plt.subplots()
    ax.plot(X, y)

    ax.set(xlabel='epoch', ylabel='bigram accuracy')

    fig.savefig("./output/bigram_accuracies.png")
    plt.show()


def kaggle(real, fake, validation):
    real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)

    validation = pd.read_csv("./data/fake_news_test_file.csv")
    validation = validation.drop(validation[validation["title"] == validation["text"]].index)
    validation["title"] = validation["title"].fillna("-NO TITLE-")
    validation["author"] = validation["author"].fillna("-NO AUTHOR-")
    validation["text"] = validation["text"].fillna("-NO TEXT-")
    validation["all_text"] = validation["title"] + " " + validation["text"]
    validation = validation.loc[:, ("id", "all_text")]

    df = pd.DataFrame(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", True), index=validation.id, columns=["label"])
    df.to_csv("./output/kaggle4.csv")


def homework(real, fake, validation):
    real_uni, real_bi, fake_uni, fake_bi = train(real, fake, False)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram", False),
                       validation.shape[0])

    bi = cal_accuracy(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", False),
                      validation.shape[0])

    print("CountVectorizer w/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train2(real, fake, False)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram", False),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", False),
                      validation.shape[0])

    print("TfidfVectorizer w/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train(real, fake, True)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram", False),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", False),
                      validation.shape[0])

    print("CountVectorizer wo/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))

    real_uni, real_bi, fake_uni, fake_bi = train2(real, fake, True)

    uni = cal_accuracy(classify(real_uni, fake_uni, validation, real.shape[0], fake.shape[0], "unigram", False),
                       validation.shape[0])
    bi = cal_accuracy(classify(real_bi, fake_bi, validation, real.shape[0], fake.shape[0], "bigram", False),
                      validation.shape[0])

    print("TfidfVectorizer wo/Stopwords\nAccuracy for unigram: {}\nAccuracy for bigram: {}".format(uni, bi))


def find_10(_real, _fake, stopword_bool):
    _real_freq = count_dict(_real["all_text"], "unigram", stopword_bool)
    _fake_freq = count_dict(_fake["all_text"], "unigram", stopword_bool)

    r_total_word_count = sum(_real_freq.values())
    f_total_word_count = sum(_fake_freq.values())
    unique_word_count = len(set(list(_real_freq.keys())) - set(list(_fake_freq.keys())))
    real_prob = _real.shape[0] / (_real.shape[0] + _fake.shape[0])
    fake_prob = _fake.shape[0] / (_real.shape[0] + _fake.shape[0])

    words = list(set(list(_real_freq.keys()) + list(_fake_freq.keys())))

    p_real_word_dict = {}
    p_fake_word_dict = {}

    p_real_word_not_dict = {}
    p_fake_word_not_dict = {}

    for word in words:
        r_count = _real_freq[word] if word in _real_freq else 0
        f_count = _fake_freq[word] if word in _fake_freq else 0

        p_word_real = (r_count + 1) / (r_total_word_count + unique_word_count)
        p_word_fake = (f_count + 1) / (f_total_word_count + unique_word_count)

        p_real_word = (p_word_real * real_prob) / ((p_word_real * real_prob)+(p_word_fake * fake_prob))
        p_fake_word = (p_word_fake * fake_prob) / ((p_word_fake * fake_prob)+(p_word_real * real_prob))

        p_real_word_not = ((1-p_word_real) * real_prob) / (((1-p_word_real) * real_prob) + ((1-p_word_fake) * fake_prob))
        p_fake_word_not = ((1-p_word_fake) * fake_prob) / (((1-p_word_fake) * fake_prob) + ((1-p_word_real) * real_prob))

        p_real_word_dict[word] = p_real_word
        p_fake_word_dict[word] = p_fake_word

        p_real_word_not_dict[word] = p_real_word_not
        p_fake_word_not_dict[word] = p_fake_word_not

    p_real_word_dict = dict(sorted(p_real_word_dict.items(), key=lambda k: k[1], reverse=True))
    p_fake_word_dict = dict(sorted(p_fake_word_dict.items(), key=lambda k: k[1], reverse=True))
    p_real_word_not_dict = dict(sorted(p_real_word_not_dict.items(), key=lambda k: k[1], reverse=True))
    p_fake_word_not_dict = dict(sorted(p_fake_word_not_dict.items(), key=lambda k: k[1], reverse=True))

    for d in [p_real_word_dict, p_real_word_not_dict, p_fake_word_dict, p_fake_word_not_dict]:
        for key in list(d.keys())[:15]:
            print("{}\t:{}".format(key, d[key]))
        print("!!!!!!!!!!!!!!!!")

def main():
    data_df = read()
    real, fake, validation = train_validation_split(df=data_df, train_ratio=0.9)
    part_1(real, fake)

    # kaggle(real, fake, validation)
    # test_randomize_data(data_df)

    # homework(real, fake, validation)
    # find_10(real, fake, False)
    # find_10(real, fake, True)