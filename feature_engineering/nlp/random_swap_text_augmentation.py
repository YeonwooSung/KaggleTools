import random


def swap_word(new_words):
    '''
    Randomly swap words in the sentence.

    Args:
        new_words (list): list of words

    Returns:
        new_words (list): list of words after swap operations
    '''
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_swap(words, n):
    '''
    Randomly swap words in the sentence n times.

    Args:
        words (list): list of words
        n (int): number of swap operations

    Returns:
        new_words (list): list of words after swap operations
    '''
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def text_aug(sentence, alpha_rs = 0.1, num_aug=3):
    '''
    Text augmentation using random swap.

    Args:
        sentence (str): sentence to be augmented
        alpha_rs (float): percentage of words in the sentence to be swapped

    Returns:
        augmented_sentences (list): list of augmented sentences
    '''
    words = sentence.split(' ')
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = num_aug

    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    return augmented_sentences


if __name__ == '__main__':
    import pandas as pd
    train = pd.read_csv('train.csv')
    aug = train['text'].apply(lambda x: text_aug(x))
