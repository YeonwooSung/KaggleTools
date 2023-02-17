import re


def preprocess_en_kr(text):
    text = re.sub(r'[^a-zA-Z0-9가-힣 ]', '', text)
    text = re.sub(r' +', ' ', text)
    return text

def preprocess_en_kr_df(df, target_key='text'):
    df[target_key] = df[target_key].apply(preprocess_en_kr)
    return df
