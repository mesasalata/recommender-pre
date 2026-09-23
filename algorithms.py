# import itertools
import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append("SelectedData")

import SelectedData.select_data as data


# ANSI escapes for utility in logging
fn = '\033[0m'
fb = '\033[1m'
fcr = '\033[31m'

tqdm_cols = 128 # number of columns in progress bars

rate_topk = True
k = 10


def average_rating(ratings):
    """Not even an algorithm. Baseline."""
    return np.average(ratings[:, 2])


def predict_rating_from_genre(ratings_user, all_ratings, movies_genres, movie_id):
    """Naive algorithm that doesn't take other users' ratings into account."""
    genre_count = len(movies_genres[0])
    sum_ratings = np.zeros(genre_count)
    total_genre_scores = np.zeros(genre_count) + 0.001
    for rating in ratings_user:
        sum_ratings += rating[2] * movies_genres[int(rating[1])]
        total_genre_scores += movies_genres[int(rating[1])]
    average_rating_vector = sum_ratings * np.reciprocal(total_genre_scores)
    predicted_score = np.divide(np.dot(average_rating_vector, movies_genres[movie_id]), np.sum(movies_genres[movie_id]))
    if not predicted_score:
        return average_rating(all_ratings[all_ratings[:, 1] == movie_id])
    return predicted_score


def predict_rating_from_genome(ratings_user, movies_genome_scores, movie_id):
    """Another naive algorithm that doesn't take other users' ratings into account."""
    sum_ratings = np.zeros(data.gene_count)
    total_genome_scores = np.zeros(data.gene_count) + 0.001
    for rating in ratings_user:
        sum_ratings += rating[2] * movies_genome_scores[int(rating[1])]
        total_genome_scores += movies_genome_scores[int(rating[1])]
    average_genome_rating_vector = sum_ratings * np.reciprocal(total_genome_scores)
    return np.divide(np.dot(average_genome_rating_vector, movies_genome_scores[movie_id]), np.sum(movies_genome_scores[movie_id]))


def predict_rating_from_genome_interest(ratings_movie, movies_genome_norm, users_interests_norm, user_id):
    """An algorithm that takes others' interest in the same movie into account."""
    movie_id = int(ratings_movie[0, 1])
    total_interest_scores = np.zeros(data.gene_count)
    vector_length = 0
    for rating in ratings_movie:
        dot = np.dot(movies_genome_norm[movie_id], users_interests_norm[int(rating[0])])
        total_interest_scores += dot * rating[2] * users_interests_norm[int(rating[0])]
        vector_length += dot
    interest_scores = total_interest_scores / vector_length # length of this vector = weighted average of all users' ratings of this movie
    return np.dot(users_interests_norm[user_id], interest_scores)


def predict_rating_from_user_interest(ratings_movie, users_interests_norm, user_id):
    """Another algorithm that takes others' interest in the same movie into account."""
    user_index = int(user_id)
    total_rating = 0
    vector_length = 0
    for rating in ratings_movie:
        dot = np.dot(users_interests_norm[user_index], users_interests_norm[int(rating[0])])
        total_rating += dot * rating[2]
        vector_length += dot
    return total_rating / vector_length # length of this vector = a different weighted average of all users' ratings of this movie


def predict_rating_from_user_neighbours(ratings_movie, user_similarities, user_id):
    """Predicts a rating from all neighboring users (who rated the same movie)."""
    users = ratings_movie[:, 0].astype(int)
    total = np.sum(np.abs(user_similarities[user_id][users]))
    if total == 0: return -1024
    return np.divide(np.dot(user_similarities[user_id][users], ratings_movie[:, 2]), total)


def predict_rating_from_movie_neighbours(ratings_user, movie_similarities, movie_id):
    """Predicts a rating from all neighboring movies (rated by the same user)."""
    movies = ratings_user[:, 1].astype(int)
    total = np.sum(movie_similarities[movie_id][movies])
    if total == 0: return -1024
    return np.divide(np.dot(movie_similarities[movie_id][movies], ratings_user[:, 2]), total)


def predict_rating_from_neighbours(ratings_movie, ratings_by_user, user_id):
    """Predicts a rating from all neighbors."""
    # Some hyperparameters
    cosine_count_threshold = 1 # Threshold for cosine count. If cosine count is too low, it means not enough similar users have been found.
    shared_movie_threshold = 1 # Threshold for shared movies. If shared movie count is too low, it means that users with high cosine values are less likely to actually be similar.
    power = 1 # The power to apply to cosines. Should be an odd number. Higher power -> more focus on a small number of similar users.

    users = ratings_movie[:, 0]
    users_reverse = reverse(users)
    user_ratings = ratings_by_user[user_id]
    user_rated_movies = user_ratings[:, 1]
    user_rating_scores = user_ratings[:, 2]
    users_cosines = np.zeros(len(users))
    for ratings in ratings_by_user:
        user = ratings[0][0]
        if user not in users: continue
        user_index = users_reverse[user]
        other_user_rated_movies = ratings[:, 1]
        # Takes advantage of how movieId is sorted in ascending order in the dataset; breaks if that changes.
        user_shared_ratings = user_rating_scores[np.isin(user_rated_movies, other_user_rated_movies)]
        if len(user_shared_ratings) < shared_movie_threshold: continue
        other_user_shared_ratings = ratings[np.isin(other_user_rated_movies, user_rated_movies), 2]
        users_cosines[user_index] = np.dot(user_shared_ratings, other_user_shared_ratings) / (np.linalg.norm(user_shared_ratings) * np.linalg.norm(other_user_shared_ratings))
    if len(users_cosines[users_cosines != 0]) <= cosine_count_threshold:
        return -1024
    # print(users_cosines[0:5])
    # print(ratings_movie[0:5, 2])
    # print(np.sum(np.abs(users_cosines)))
    users_cosines = np.pow(users_cosines, power)
    return np.divide(np.dot(users_cosines, ratings_movie[:, 2]), np.sum(np.abs(users_cosines)))


##
## Utility + Display
##


def std_transform_inplace(user_ratings, gen): # Map separately for positive and negative values around standard deviation? Mean rating tends to be above 2.75
    scores = user_ratings[:, 2]
    std = np.std(scores)
    if std == 0:
        std = 1
    mean = np.mean(scores)
    user_ratings[:, 2] = (scores - mean + gen.choice([-0.001, 0.001], len(scores))) / std
    # user_ratings[:, 2] = (scores - 0.5 + 0.001) / std
    # user_ratings[:, 2] -= 3.65


# def std_transform_inplace(user_ratings):
#     scores = user_ratings[:, 2]
#     if all(scores == scores[0]):
#         user_ratings[:, 2] = np.random.choice([-0.001, 0.001], len(scores)) # Balanced distribution on both sides of 0
#         return
#     mean = np.median(scores)
#     scores_pos_filter = scores >= mean
#     std_pos = np.std(np.concat((scores[scores_pos_filter], [mean])))
#     scores_neg_filter = scores <= mean
#     std_neg = np.std(np.concat((scores[scores_neg_filter], [mean])))
#     if any(std_pos * scores_pos_filter + std_neg * scores_neg_filter == 0):
#         print(std_pos * scores_pos_filter + std_neg * scores_neg_filter)
#         print(scores)
#         print(mean)
#         print(std_pos)
#         print(std_neg)
#         print(scores[scores_pos_filter])
#         print(scores[scores_neg_filter])
#         print()
#     user_ratings[:, 2] -= mean - np.random.choice([-0.001, 0.001], len(scores))
#     user_ratings[:, 2] /= std_pos * scores_pos_filter + std_neg * scores_neg_filter


def cossim(vector1, vector2):
    """Calculates cosine similarity."""
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if not norm: return 0 # For safety
    return np.dot(vector1, vector2) / norm


def reverse(l):
    return {p: i for i, p in enumerate(l)}


def diff_4(a, b): return np.pow(a - b, 4)
def diff_4_aggr(a): return np.pow(a, 0.25)
def diff_2(a, b): return (a - b) ** 2
def diff_2_aggr(a): return np.pow(a, 0.5)
def diff_abs(a, b): return np.abs(a - b)
def diff(a, b): return a - b
def sign_diff(a, b): return np.sign(a) != np.sign(b)
def identity(a): return a
aggr = identity


def calc_error(l1, l2, ef=diff_2, af=diff_2_aggr):
    """
    Aggregates an error value from two arrays (typically predicted and expected value arrays).
    :param l1: Array 1
    :param l2: Array 2
    :param ef: Error function (applied on each pair of values in l1 and l2)
    :param af: Aggregation function (applied on the mean of the error function)
    :return:
    """
    if not len(l1): return 0
    errors = np.vectorize(ef)(l1, l2)
    error = af(np.mean(errors))
    return error


def display_error(predicted, actual, plot=True, show=True, algorithm="", ef=diff_2, af=diff_2_aggr, clustered=False):
    algorithm_suffix = int(bool(len(algorithm))) * f" ({algorithm})" # Jank

    if plot: # Plot zero rating performance?
        # Plotting
        bar_xs = np.array(range(-50, 52)) / 10.0
        bar_xs_reverse = reverse(bar_xs)
        bar_predicted_counts = np.zeros(len(bar_xs))
        predicted_by_x = [[] for _ in range(len(bar_xs))]
        for i, score in enumerate(predicted):
            x = np.round(score * 10) / 10
            if x > 5.0 or x < -5.0: x = 5.1
            x_index = bar_xs_reverse[x]
            bar_predicted_counts[x_index] += 1
            predicted_by_x[x_index].append(i)
        bar_actual_counts = np.zeros(len(bar_xs))
        actual_by_x = [[] for _ in range(len(bar_xs))]
        for i, score in enumerate(actual):
            x = np.round(score * 10) / 10
            if x > 5.0 or x < -5.0: x = 5.1
            x_index = bar_xs_reverse[x]
            bar_actual_counts[bar_xs_reverse[x]] += 1
            actual_by_x[x_index].append(i)
        error_by_x_by_predicted_x = np.array([calc_error(predicted[x], actual[x], ef=ef, af=af) for x in predicted_by_x])
        error_by_x_by_actual_x = np.array([calc_error(predicted[x], actual[x], ef=ef, af=af) for x in actual_by_x])
        # error_by_x_zero_rating = np.array([calc_error(np.zeros(len(x)) + 0.001, actual[x], ef=ef, af=af) for x in actual_by_x])
        error_by_x_zero_rating = np.array([calc_error(np.random.choice([-0.001, 0.001], len(x)), actual[x], ef=ef, af=af) for x in actual_by_x])

        bar_xs_fine = np.array(range(-500, 502)) / 100.0
        bar_xs_fine_reverse = reverse(bar_xs_fine)
        bar_predicted_counts_fine = np.zeros(len(bar_xs_fine))
        for score in predicted:
            x = np.round(score * 100) / 100
            if x > 5.00 or x < -5.00: x = 5.01
            x_index = bar_xs_fine_reverse[x]
            bar_predicted_counts_fine[x_index - 5:x_index + 5] += 1

        fig1, x1 = plt.subplots()
        # x1.plot(bar_xs[:-1], bar_predicted_counts[:-1], color='#0000ff', label="Predicted Scores")
        x1.plot(bar_xs_fine[:-1], bar_predicted_counts_fine[:-1], color='#0000ff', label="Predicted Scores")
        x1.plot(bar_xs[:-1], bar_actual_counts[:-1], color='#00ff00', label="Actual Scores")
        x1.legend()
        x1.set_xticks(np.array(range(-10, 11)) / 2.0)
        x1.grid(True)
        x1.set_xlabel("Rating")
        x1.set_ylabel("Count")
        x1.set_title(f"Distribution{algorithm_suffix}")

        fig2, x2 = plt.subplots()
        error_by_x_by_predicted_x_filter = error_by_x_by_predicted_x[:-1] > 0
        error_by_x_by_actual_x_filter = error_by_x_by_actual_x[:-1] > 0
        if not clustered: x2.plot(bar_xs[:-1][error_by_x_by_predicted_x_filter], error_by_x_by_predicted_x[:-1][error_by_x_by_predicted_x_filter], color='#0000ff', label="Error by Predicted Scores")
        x2.plot(bar_xs[:-1][error_by_x_by_actual_x_filter], error_by_x_by_actual_x[:-1][error_by_x_by_actual_x_filter], color='#00ff00', label="Error by Actual Scores")
        x2.plot(bar_xs[:-1][error_by_x_by_actual_x_filter], error_by_x_zero_rating[:-1][error_by_x_by_actual_x_filter], color='#ff00ff', label="Error by Actual Scores (Zero Rating)")
        x2.legend()
        x2.set_xticks(np.array(range(-10, 11)) / 2.0)
        x2.grid(True)
        x2.set_xlabel("Rating")
        x2.set_ylabel("Error")
        x2.set_title(f"Errors by Rating{algorithm_suffix}")

        if show: plt.show()


def format_algorithm_name(name: str):
    return name.lower().replace(' ', '-')


def report_error_from_cache(algorithm: str, cluster: float | None = None, scaling=1, plot=True, show=True, ef=diff_2, af=diff_2_aggr):
    predicted = data.get_scores(file=f"{format_algorithm_name(algorithm)}-cache-predicted.csv")
    actual = data.get_scores(file=f"{format_algorithm_name(algorithm)}-cache-actual.csv")
    report_error(predicted, actual, algorithm, cluster=cluster, scaling=scaling, plot=plot, show=show, ef=ef, af=af, cached=True)


def report_error(
        predicted=None,
        actual=None,
        algorithm: str = "",
        cluster: float | None = None,
        scaling=1, plot=True,
        show=False,
        ef=diff_2,
        af=diff_2_aggr,
        cached=False,
        algorithm_func=None,
        **kwargs
    ):
    if algorithm_func is not None:
        predicted = algorithm_func(kwargs)
    elif predicted is None:
        print(f"{fcr}{fb}(report_error) No predicted values or prediction function provided!{fn}")
        return
    try:
        if test_set is not None:
            actual = test_set[:, 2]
        elif actual is None:
            print(f"{fcr}{fb}(report_error) No actual values or test set provided!{fn}")
            return
    except NameError:
        if actual is None:
            print(f"{fcr}{fb}(report_error) No actual values or test set provided!{fn}")
            return

    if not cached:
        data.save_scores(f"{format_algorithm_name(algorithm)}-cache-predicted.csv", predicted)
        data.save_scores(f"{format_algorithm_name(algorithm)}-cache-actual.csv", actual)

    predicted_mask = (-5 < predicted) * (predicted < 5)
    predicted_masked = predicted[predicted_mask] * scaling
    actual_masked = actual[predicted_mask]
    fail_rate_suffix = f", fail rate: {fb}{100 * (1 - len(predicted_masked) / len(predicted)):.2f}%{fn}"
    selection_suffix = ""

    if cluster is not None:
        old_predicted_masked_len = len(predicted_masked)
        predicted_mask2 = abs(predicted_masked - cluster) < 0.05
        predicted_masked = predicted_masked[predicted_mask2]
        actual_masked = actual_masked[predicted_mask2]
        selection_suffix = f", selected: {fb}{100 * (len(predicted_masked) / old_predicted_masked_len):.2f}%{fn} ({len(predicted_masked)} of {old_predicted_masked_len})"
    # if training_sets is not None and test_set is not None:
    #     # Constant
    #     k = 10
    #     count = 10
    #
    #     random.seed(69420)
    #     random_users = np.array([random.randint(0, user_count - 1) for _ in range(10)])
    #     random_users = random_users[len(training_sets[random_users]) >= k * 2]
    #     topk_training_sets = []
    #     topk_test_set = []
    #     for user in random_users:
    #         sort = sorted(training_sets[user], key=lambda x: x[2], reverse=True)
    #         topk_test_set += sort[:10]
    #         topk_training_sets.append(np.array(sort[10:]))
    #     topk_predicted =

    display_error(predicted_masked, actual_masked, plot=plot, show=show, algorithm=algorithm, ef=ef, af=af, clustered=cluster is not None)

    predicted_rating_error_sq = calc_error(predicted_masked, actual_masked, ef, af)
    predicted_rating_error = calc_error(predicted_masked, actual_masked, diff_abs, aggr)
    predicted_rating_bias = calc_error(predicted_masked, actual_masked, diff, aggr)
    predicted_sign_accuracy = 100 - calc_error(predicted_masked, actual_masked, sign_diff, identity) * 100
    zero_sign_accuracy = 100 - calc_error(np.zeros(len(predicted_masked)) + 0.001, actual_masked, sign_diff, identity) * 100
    print(f"Average error ({fb}{algorithm}{fn} algorithm): {fb}{predicted_rating_error_sq:.4f}{fn}. (MAE: {fb}{predicted_rating_error:.4f}{fn}, bias: {fb}{predicted_rating_bias:.4f}{fn}{fail_rate_suffix}{selection_suffix}, sign accuracy: {fb}{predicted_sign_accuracy:.2f}{fn}% (baseline: {fb}{zero_sign_accuracy:.2f}{fn}%))")
    # print(np.average([abs(predicted_masked[i] - (5.5 / 2)) - abs(actual_masked[i] - (5.5 / 2)) for i in range(predicted_masked_count)]))


##
## Testing
##


def test_average_rating(_kwargs=None):
    """Average movie rating as estimator function."""
    movie_subsets = [[] for _ in range(movie_count)]
    for rating in dataset:
        movie_subsets[int(rating[1])].append(rating)
    movie_subsets = [np.array(subset) for subset in movie_subsets]
    movie_average_ratings = np.array([average_rating(subset) if len(subset) else 0 for subset in movie_subsets])
    movie_training_subsets = []
    for rating in test_set:
        movie_training_subsets.append(movie_subsets[int(rating[1])])

    if rate_topk:
        # Top K calculations (was put here because of performance cost otherwise)
        # Recall @ K
        topk_accuracy_total = 0
        topk_ratio_total = 0
        topk_count = 0
        for user_ratings in training_sets:
            if len(user_ratings) < k * 2: continue # Skip over too-small lists # or random.random() > 0.1 # and only consider 10% of users (for performance)
            sort = np.array(sorted(user_ratings, key=lambda x: x[2], reverse=True))
            rated_movies = sort[:, 1].astype(int)
            user_predicted_ratings_indices = np.argsort(movie_average_ratings[rated_movies])
            topk_accuracy_total += np.sum(0 in user_predicted_ratings_indices[-k:]) # Recall top 1 in top K
            # topk_accuracy_total += np.sum(user_predicted_ratings_indices[-k:] < k) / k # Recall top K in top K (-> intersection / K)
            topk_ratio_total += k / float(len(user_ratings))
            topk_count += 1
        topk_accuracy = topk_accuracy_total / topk_count
        topk_ratio = topk_ratio_total / topk_count
        print(f"Recall @ K(={k}) (Matrix SGD algorithm): {round(topk_accuracy * 1000) / 10.0}% (baseline = {round(topk_ratio * 1000) / 10.0}%)")

        # Precision @ K
        topk_precision_total = 0
        relevant_ratio_total = 0
        topk_count = 0
        for user_ratings in training_sets:
            if len(user_ratings) < k * 2: continue # Skip over too-small lists # or random.random() > 0.1 # and only consider 10% of users (for performance)
            rated_movies = user_ratings[:, 1].astype(int)
            user_predicted_ratings_indices = np.argsort(movie_average_ratings[rated_movies])
            top_k_values = user_ratings[user_predicted_ratings_indices[-10:], 2]
            topk_precision_total += np.sum(top_k_values > 0) / k
            relevant_ratio_total += np.sum(user_ratings[:, 2] > 0) / len(user_ratings)
            topk_count += 1
        topk_accuracy = topk_precision_total / topk_count
        relevant_ratio = relevant_ratio_total / topk_count
        print(f"Precision @ K(={k}) (Matrix SGD algorithm): {round(topk_accuracy * 1000) / 10.0}% (baseline = {round(relevant_ratio * 1000) / 10.0}%)")

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([average_rating(subset) if len(subset) else 0 for subset in tqdm(movie_training_subsets, ncols=tqdm_cols)])
    return user_predicted_ratings # report_error(user_predicted_ratings, test_set[:, 2], "Average Rating")


def test_rating_from_genre(movies_genres):
    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_genre(training_sets[i], dataset, movies_genres, int(test_set[i, 1])) for i in tqdm(range(len(test_set)), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "Naive Genre Rating")


def test_rating_from_genome(movies_genome_scores):
    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_genome(training_sets[i], movies_genome_scores, int(test_set[i, 1])) for i in tqdm(range(len(test_set)), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "Naive Genome Rating")


def test_rating_naive_hybrid(movies_genres, movies_genome_scores, movies_genome_norm, users_interests_norm):
    test_count = len(test_set)

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings_genre = np.array([predict_rating_from_genre(training_sets[i], dataset, movies_genres, int(test_set[i, 1])) for i in tqdm(range(test_count), ncols=tqdm_cols)])
    user_predicted_ratings_genome = np.array([predict_rating_from_genome(training_sets[i], movies_genome_scores, int(test_set[i, 1])) for i in tqdm(range(test_count), ncols=tqdm_cols)])
    print("Calculating blending factors...")
    movies_genres_norm = np.array([movie_genres / np.linalg.norm(movie_genres) for movie_genres in movies_genres])
    users_genre_interests = np.zeros((user_count, len(movies_genres[0])))
    time.sleep(0.02)
    for review in tqdm(dataset, ncols=tqdm_cols):
        users_genre_interests[int(review[0])] += movies_genres_norm[int(review[1])] * review[2]
    users_genre_interests_norm = [np.divide(genre_interests, np.linalg.norm(genre_interests)) for genre_interests in users_genre_interests]
    users_predicted_genre_interests = np.array([np.dot(users_genre_interests_norm[i], movies_genres_norm[int(test_set[i, 1])]) for i in range(test_count)])
    users_predicted_genome_interests = np.array([np.dot(users_interests_norm[i], movies_genome_norm[int(test_set[i, 1])]) for i in range(test_count)])
    blending_factors = np.array([genome_interests / (genre_interests + genome_interests) for genre_interests, genome_interests in zip(users_predicted_genre_interests, users_predicted_genome_interests)])
    report_error((user_predicted_ratings_genre * (1 - blending_factors) + user_predicted_ratings_genome * blending_factors), test_set[:, 2], "Naive Hybrid Rating")


def test_rating_from_genome_interest(movies_genome_norm, users_interests_norm):
    test_count = len(test_set)

    movie_subsets = [[] for _ in range(movie_count)]
    for rating in dataset:
        movie_subsets[int(rating[1])].append(rating)
    movie_subsets = [np.array(subset) for subset in movie_subsets]
    movie_training_subsets = []
    for rating in test_set:
        if rating[1] > movie_count - 10: print(rating[1])
        movie_training_subsets.append(movie_subsets[int(rating[1])])

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_genome_interest(movie_training_subsets[i], movies_genome_norm, users_interests_norm, int(test_set[i, 0])) if len(movie_training_subsets[i]) else -1024 for i in tqdm(range(test_count), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "Genome Interest")


def test_rating_from_user_interest(users_interests_norm):
    test_count = len(test_set)

    movie_subsets = [[] for _ in range(movie_count)]
    for rating in dataset:
        movie_subsets[int(rating[1])].append(rating)
    movie_subsets = [np.array(subset) for subset in movie_subsets]
    movie_training_subsets = []
    for rating in test_set:
        if rating[1] > movie_count - 10: print(rating[1])
        movie_training_subsets.append(movie_subsets[int(rating[1])])

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_user_interest(movie_training_subsets[i], users_interests_norm, int(test_set[i, 0])) if len(movie_training_subsets[i]) else -1024 for i in tqdm(range(test_count), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "User Interest")


def test_rating_from_user_neighbours(user_similarities):
    test_count = len(test_set)

    movie_subsets = [[] for _ in range(movie_count)]
    for rating in dataset:
        movie_subsets[int(rating[1])].append(rating)
    movie_subsets = [np.array(subset) for subset in movie_subsets]
    movie_training_subsets = []
    for rating in test_set:
        if rating[1] > movie_count - 10: print(rating[1])
        movie_training_subsets.append(movie_subsets[int(rating[1])])

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_user_neighbours(movie_training_subsets[i], user_similarities, int(test_set[i, 0])) if len(movie_training_subsets[i]) else -1024 for i in tqdm(range(test_count), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "User Neighbour")


def test_rating_from_movie_neighbours(movie_similarities):
    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_movie_neighbours(training_sets[i], movie_similarities, int(test_set[i, 1])) if len(training_sets[i]) else -1024 for i in tqdm(range(len(test_set)), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, test_set[:, 2], "Movie Neighbour")


def test_rating_from_neighbours():
    all_movies = np.unique(dataset[:, 1])
    all_movies_reverse = reverse(all_movies)
    all_users = np.unique(dataset[:, 0])
    all_users_reverse = reverse(all_users)

    user_subsets = [[] for _ in range(user_count)]
    for rating in dataset:
        user_subsets[all_users_reverse[rating[0]]].append(rating)
    user_subsets = [np.array(subset) for subset in user_subsets]
    movie_subsets = [[] for _ in range(movie_count)]
    movie_test_users = np.zeros(movie_count)
    movie_test_ratings = np.zeros(movie_count)
    for rating in dataset:
        movie_subsets[all_movies_reverse[rating[1]]].append(rating)
    for i, subset in enumerate(movie_subsets):
        if not len(subset): continue
        movie_test = movie_subsets[i].pop(random.randint(0, len(subset) - 1))
        movie_test_users[i] = movie_test[0]
        movie_test_ratings[i] = movie_test[2]
    movie_subsets = [np.array(subset) for subset in movie_subsets]

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([predict_rating_from_neighbours(movie_subsets[i], user_subsets, int(movie_test_users[i])) if len(movie_subsets[i]) else -1024 for i in tqdm(range(0, movie_count, 20), ncols=tqdm_cols)])
    report_error(user_predicted_ratings, movie_test_ratings[range(0, movie_count, 20)], "Neighbour Predictor")


def test_rating_from_matrix(_kwargs=None):
    # Hyperparameter(s)
    factor_count = 50
    epochs = 20
    learning_rate = 0.005
    regularization_rate = 0.02
    dist_scale = 0.1

    epoch_print_interval = 1

    gen = np.random.default_rng(42069)
    user_vectors = gen.normal(0.0, dist_scale, (user_count, factor_count))
    movie_vectors = gen.normal(0.0, dist_scale, (movie_count, factor_count))

    print("Converting data...")
    time.sleep(0.02)
    bar = tqdm(total=3, ncols=tqdm_cols)
    dataset_users = dataset[:, 0].astype(int)
    bar.update(1)
    dataset_movies = dataset[:, 1].astype(int)
    bar.update(1)
    dataset_scores = dataset[:, 2]
    bar.update(1)
    bar.close()
    time.sleep(0.02)

    print(f"Training matrix ({epochs} epochs (* {len(dataset_users)} iterations per epoch), {factor_count} factor size)...")
    time.sleep(0.02)
    for e in tqdm(range(epochs), ncols=tqdm_cols, unit="ep"):
        if e % epoch_print_interval == 0:
            time.sleep(0.02)
            print()
            print(f"Epoch {e + 1}.")
            user_predicted_ratings = np.array([np.dot(user_vectors[int(test_rating[0])], movie_vectors[int(test_rating[1])]) for test_rating in test_set])
            print(f"RMSE @ {e} epochs: {np.sqrt(np.mean((user_predicted_ratings - test_set[:, 2]) ** 2))}")
            time.sleep(0.02)
        for user, movie, score in zip(dataset_users, dataset_movies, dataset_scores): # was random selection
            error = score - np.dot(user_vectors[user], movie_vectors[movie])
            user_vector = user_vectors[user]
            user_vectors[user] += learning_rate * (error * movie_vectors[movie] - regularization_rate * user_vector)
            movie_vectors[movie] += learning_rate * (error * user_vector - regularization_rate * movie_vectors[movie])

    if rate_topk:
        # Top K calculations (was put here because of performance cost otherwise)
        # Recall @ K
        topk_accuracy_total = 0
        topk_ratio_total = 0
        topk_count = 0
        for user_ratings in training_sets:
            if len(user_ratings) < k * 2: continue # Skip over too-small lists # or random.random() > 0.1 # and only consider 10% of users (for performance)
            user = int(user_ratings[0][0])
            sort = np.array(sorted(user_ratings, key=lambda x: x[2], reverse=True))
            rated_movies = sort[:, 1].astype(int)
            user_predicted_ratings_indices = np.argsort(np.dot(user_vectors[user], np.transpose(movie_vectors[rated_movies])))
            topk_accuracy_total += np.sum(0 in user_predicted_ratings_indices[-k:]) # Recall top 1 in top K
            # topk_accuracy_total += np.sum(user_predicted_ratings_indices[-k:] < k) / k # Recall top K in top K (-> intersection / K)
            topk_ratio_total += k / float(len(user_ratings))
            topk_count += 1
        topk_accuracy = topk_accuracy_total / topk_count
        topk_ratio = topk_ratio_total / topk_count
        print(f"Recall @ K(={k}) (Matrix SGD algorithm): {round(topk_accuracy * 1000) / 10.0}% (baseline = {round(topk_ratio * 1000) / 10.0}%)")

        # Precision @ K
        topk_precision_total = 0
        relevant_ratio_total = 0
        topk_count = 0
        for user_ratings in training_sets:
            if len(user_ratings) < k * 2: continue # Skip over too-small lists # or random.random() > 0.1 # and only consider 10% of users (for performance)
            user = int(user_ratings[0][0])
            rated_movies = user_ratings[:, 1].astype(int)
            user_predicted_ratings_indices = np.argsort(np.dot(user_vectors[user], np.transpose(movie_vectors[rated_movies])))
            top_k_values = user_ratings[user_predicted_ratings_indices[-10:], 2]
            topk_precision_total += np.sum(top_k_values > 0) / k
            relevant_ratio_total += np.sum(user_ratings[:, 2] > 0) / len(user_ratings)
            topk_count += 1
        topk_accuracy = topk_precision_total / topk_count
        relevant_ratio = relevant_ratio_total / topk_count
        print(f"Precision @ K(={k}) (Matrix SGD algorithm): {round(topk_accuracy * 1000) / 10.0}% (baseline = {round(relevant_ratio * 1000) / 10.0}%)")

    print("Getting predicted ratings...")
    time.sleep(0.02)
    user_predicted_ratings = np.array([np.dot(user_vectors[int(test_set[i, 0])], movie_vectors[int(test_set[i, 1])]) for i in tqdm(range(len(test_set)), ncols=tqdm_cols)])
    return user_predicted_ratings


dataset: np.ndarray
training_sets: list[np.ndarray]
test_set: np.ndarray
user_count: int
movie_count: int


def main() -> None:
    global rate_topk, k
    global dataset, training_sets, test_set, user_count, movie_count

    print("Loading samples...")
    sample_all = data.get_rating_samples(file="SelectedData/sample-all.csv")
    # sample_all = np.array([
    #     [0.0, 0.0, 1.5],
    #     [0.0, 1.0, 5.0],
    #     [0.0, 2.0, 3.0],
    #     [1.0, 0.0, 4.5],
    #     [1.0, 1.0, 3.5],
    #     [1.0, 3.0, 0.5],
    #     [3.0, 0.0, 4.0],
    #     [3.0, 2.0, 4.0],
    #     [3.0, 3.0, 4.0]
    # ])
    # sample_all = sample_all_unfiltered[:10000]
    # rate_topk = False
    # sample_all_test = data.get_rating_samples(file="SelectedData/sample-all-test.csv")

    print("Loading movie genres...")
    movies_genres = data.get_movie_genres()

    print("Loading genome scores...")
    movies_genomes = data.get_movie_genome_scores()
    movies_genome_keys = np.array(list(movies_genomes.keys()))

    print("Filtering data...")
    sample_all_filter = np.isin(sample_all[:, 1], movies_genome_keys)
    sample_all_filtered = sample_all[sample_all_filter]
    sample_all_filter2 = [True for _ in range(len(sample_all_filtered))]
    count = 0
    current_user = 0
    for i, rating in enumerate(sample_all_filtered):
        if rating[0] != current_user:
            if count == 1:
                sample_all_filter2[i - 1] = False
            current_user = rating[0]
            count = 0
        count += 1
    if count == 1: sample_all_filter2[-1] = False
    sample_all_filtered = sample_all_filtered[sample_all_filter2]
    all_users = np.unique(sample_all_filtered[:, 0])
    all_users_reverse = reverse(all_users)
    user_count = len(all_users)

    print("Transforming data...")
    all_movies = np.unique(sample_all_filtered[:, 1])
    all_movies_reverse = reverse(all_movies)
    movie_count = len(all_movies)
    user_subsets = [[] for _ in range(user_count)]
    time.sleep(0.02)
    for rating in tqdm(sample_all_filtered, ncols=tqdm_cols):
        rating[0] = all_users_reverse[rating[0]] # Collapse users to indices
        rating[1] = all_movies_reverse[rating[1]] # Collapse movies to indices
        user_subsets[int(rating[0])].append(rating)
    user_subsets = [np.array(subset) for subset in user_subsets]
    gen = np.random.default_rng(200100)
    for subset in user_subsets:
        std_transform_inplace(subset, gen) # User bias removed
        # Note that movie bias (the skew of the average rating given to a movie) is NOT removed
    movies_genres = np.array([movies_genres[key] for key in movies_genres.keys() if key in movies_genome_keys]) # Collapse to indices
    movies_genomes = np.array(list(movies_genomes.values())) # Collapse to indices

    print("Preparing testing data...")
    user_training_sets = []
    user_test_set = []
    random.seed(1337)
    for subset in user_subsets:
        test_rating = subset[random.randint(0, len(subset) - 1)]
        user_training_sets.append(subset[subset[:, 1] != test_rating[1]])
        user_test_set.append(test_rating)
    user_training_sets = [np.array(subset) for subset in user_training_sets]
    user_test_set = np.array(user_test_set)
    sample_all_transformed = np.concat(user_training_sets)

    # Debug prints
    # mv = np.unique(sample_all_transformed[:, 1])
    # print(mv)
    # print(len(mv))
    # print(mv[np.isin(mv, list(range(movie_count)))])
    # print(len(mv[np.isin(mv, list(range(movie_count)))]))
    # mv = np.unique(user_test_set[:, 1])
    # print(mv)
    # print(len(mv))
    # print(mv[np.isin(mv, list(range(movie_count)))])
    # print(len(mv[np.isin(mv, list(range(movie_count)))]))

    # print("Computing genome normalizations...")
    # movies_genome_norm = [np.divide(genome, np.linalg.norm(genome)) for genome in movies_genomes]
    #
    # print("Computing interests...")
    # users_interests = np.zeros((user_count, data.gene_count))
    # time.sleep(0.02)
    # for review in tqdm(sample_all_transformed, ncols=tqdm_cols):
    #     users_interests[int(review[0])] += movies_genome_norm[int(review[1])] * review[2]
    # time.sleep(0.02)
    # users_interests_norm = [np.divide(user_interests, np.linalg.norm(user_interests)) for user_interests in tqdm(users_interests, ncols=tqdm_cols)]

    # Unweighted interests - Confirmed to be worse (Genome Interest RMSE increased by ~0.02)
    # users_interests2 = np.zeros((user_count, data.gene_count))
    # time.sleep(0.02)
    # for review in tqdm(sample_all_transformed, ncols=tqdm_cols):
    #     users_interests2[int(review[0])] += movies_genome_norm[int(review[1])]
    # time.sleep(0.02)
    # users_interests2_norm = [np.divide(user_interests, np.linalg.norm(user_interests)) for user_interests in tqdm(users_interests2, ncols=tqdm_cols)]

    # Cache regeneration:
    # print("Producing user similarity matrix...")
    # movie_subsets = [[] for _ in range(movie_count)]
    # for rating in sample_all_transformed:
    #     movie_subsets[int(rating[1])].append(rating)
    # movie_subsets = [np.array(subset) for subset in movie_subsets]
    # movie_subsets_rating = [subset[:, 2] if len(subset) else [] for subset in movie_subsets]
    # movie_subsets_user = [subset[:, 0].astype(int) if len(subset) else [] for subset in movie_subsets]
    # user_similarities = np.zeros((user_count, user_count)) # x = movie1, y = movie2, lookup with y > x
    # user1_length_sqs = np.zeros((user_count, user_count))
    # user2_length_sqs = np.zeros((user_count, user_count))
    # time.sleep(0.02)
    # for i in tqdm(range(len(movie_subsets)), ncols=tqdm_cols):
    #     ratings = movie_subsets_rating[i]
    #     users = movie_subsets_user[i]
    #     # for a, b in itertools.combinations(range(len(users)), 2):
    #     #     coord = users[[a, b]]
    #     #     user_similarities[coord] += ratings[a] * ratings[b]
    #     #     user1_length_sqs[coord] += ratings[a] ** 2
    #     #     user2_length_sqs[coord] += ratings[b] ** 2
    #     for (rating1, rating2), (i1, i2) in zip(itertools.combinations(ratings, 2), itertools.combinations(users, 2)):
    #         user_similarities[i1, i2] += rating1 * rating2
    #         user1_length_sqs[i1, i2] += rating1 ** 2
    #         user2_length_sqs[i1, i2] += rating2 ** 2
    # for i, row in enumerate(user_similarities):
    #     dot_filter = row != 0.0
    #     user_similarities[i][dot_filter] /= np.sqrt(user1_length_sqs[i][dot_filter] * user2_length_sqs[i][dot_filter])
    # user_similarities += np.transpose(user_similarities)
    # data.save_matrix("user-similarity-cache.csv", user_similarities)

    # Cache retrieval:
    # print("Loading user similarity matrix...")
    # user_similarities = data.get_matrix("user-similarity-cache.csv")

    # Cache regeneration:
    # print("Producing movie similarity matrix...")
    # movie_similarities = np.zeros((movie_count, movie_count)) # x = movie1, y = movie2, lookup with y > x
    # movie1_length_sqs = np.zeros((movie_count, movie_count))
    # movie2_length_sqs = np.zeros((movie_count, movie_count))
    # time.sleep(0.02)
    # for ratings in tqdm(user_training_sets, ncols=tqdm_cols):
    #     for rating1, rating2 in itertools.combinations(ratings, 2):
    #         coord = int(rating1[1]), int(rating2[1])
    #         movie_similarities[coord] += rating1[2] * rating2[2]
    #         movie1_length_sqs[coord] += rating1[2] ** 2
    #         movie2_length_sqs[coord] += rating2[2] ** 2
    # for i, row in enumerate(movie_similarities):
    #     dot_filter = row != 0.0
    #     movie_similarities[i][dot_filter] /= np.sqrt(movie1_length_sqs[i][dot_filter] * movie2_length_sqs[i][dot_filter])
    # movie_similarities += np.transpose(movie_similarities)
    # data.save_matrix("movie-similarity-cache.csv", movie_similarities)

    # Cache retrieval:
    # print("Loading movie similarity matrix...")
    # movie_similarities = data.get_matrix("movie-similarity-cache.csv")

    print("Setting globals...")
    dataset = sample_all_transformed
    training_sets = user_training_sets
    test_set = user_test_set

    print("Preparation complete.")
    print()

    # Tests
    report_error(algorithm="Zero Rating", algorithm_func=lambda _: np.zeros(len(test_set)) + 0.001) # -0.001 + (random.random() > 0.5) * 0.002
    print()
    report_error(algorithm="Average Rating", algorithm_func=test_average_rating)
    print()
    # test_rating_from_genre(sample_all_transformed, user_training_sets, user_test_set, movies_genres)
    # print()
    # test_rating_from_genome(user_training_sets, user_test_set, movies_genomes)
    # print()
    # test_rating_naive_hybrid(sample_all_transformed, user_training_sets, user_test_set, user_count, movies_genres, movies_genomes, movies_genome_norm, users_interests_norm)
    # print()
    # test_rating_from_genome_interest(sample_all_transformed, user_test_set, movie_count, movies_genome_norm, users_interests_norm)
    # print()
    # test_rating_from_user_interest(sample_all_transformed, user_test_set, movie_count, users_interests_norm)
    # print()
    # test_rating_from_user_neighbours(sample_all_transformed, user_test_set, movie_count, user_similarities)
    # print()
    # test_rating_from_movie_neighbours(user_training_sets, user_test_set, movie_similarities)
    # print()
    # test_rating_from_neighbours(sample_all_transformed, user_count, movie_count)
    # print()
    report_error(algorithm="Matrix SGD", algorithm_func=test_rating_from_matrix)

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()
