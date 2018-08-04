import email
import os
import sys
from random import sample
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# ------------------------------------------------------ #

# Main directory of corpus
RESOURCES_DIR = "Resources/publiccorpus/"

# Categories
CATEGORY_MAP = {
    'easy_ham': 0, 'easy_ham_2': 0, 'hard_ham': 0, 'spam': 1, 'spam_2': 1
}
CATEGORY_NAMES = ['ham', 'spam']

# Setting sizes of training and testing set
ALL = -1                            # used to indicate a maximum test set size
TRAIN_PER_CAT = int(sys.argv[1])    # training samples from each category
TESTS_PER_CAT = ALL                 # testing samples from each category

# Lists for training and testing msgs and their categories
train_strings = []   # list of training messages
tests_strings = []   # list of testing messages
train_cats = []      # corresponding categories of training messages
tests_cats = []      # corresponding categories of testing messages

# Count vectorizer and tf-idf transformer
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# ------------------------------------------------------ #

# Converts a message to a string
# - Plain parts are converted to lowercase
# - Html  parts are converted to "html"
# - Other parts are ignored
def msg_to_string(msg):
    if msg.is_multipart():  # convert each sub-part and join all strings together
        text_list = list(map(msg_to_string, msg.get_payload()))
        return (" ").join(text_list)
    elif msg.get_content_type() == 'text/plain':
        return msg.get_payload().lower()
    elif msg.get_content_type() == 'text/html':
        return "html"  # str(msg.get_payload()).replace("</", " ").replace("<", " ").replace(">", " ").lower() + " "
    else:
        return ""


# List to keep track of no. of train/test examples collected for each category
train_counts = [0] * len(CATEGORY_NAMES)
tests_counts = [0] * len(CATEGORY_NAMES)

# Converts each message in a folder to a string and adds to training/testing sets
def convert_and_add_files(folder_name):
    # Obtain category of folder and folder's directory
    folder_cat = CATEGORY_MAP[folder_name]
    folder_dir = os.path.join(RESOURCES_DIR, folder_name)

    # Go through all files randomly and fill up training and testing sets
    files_list = os.listdir(folder_dir)
    for file in sample(files_list, len(files_list)):

        # Open message file and convert the message to a string
        with open(os.path.join(folder_dir, file), 'r', encoding='ISO-8859-1') as input_file:
            msg_text = msg_to_string(email.message_from_file(input_file))
            input_file.close()

        # If training set filled up, start filling up testing set
        if train_counts[folder_cat] < TRAIN_PER_CAT:
            train_strings.append(msg_text)
            train_cats.append(folder_cat)
            train_counts[folder_cat] += 1
        elif tests_counts[folder_cat] < TESTS_PER_CAT or TESTS_PER_CAT == ALL:
            tests_strings.append(msg_text)
            tests_cats.append(folder_cat)
            tests_counts[folder_cat] += 1
        else:
            break  # training and testing set filled up

# For each folder in main directory, convert all files in that folder
print("Going through folders...")
for folder in CATEGORY_MAP.keys():
    convert_and_add_files(folder)
    print(folder + "...DONE.")

# Total actual size of training and testing sets
total_train = len(train_strings)
total_tests = len(tests_strings)

# ------------------------------------------------------- #

# For training set: obtain bag of counts and apply tfidf
X_train = count_vectorizer.fit_transform(train_strings)
X_train = tfidf_transformer.fit_transform(X_train)

# For testing set: obtain bag of counts and apply tfidf
X_tests = count_vectorizer.transform(tests_strings)
X_tests = tfidf_transformer.transform(X_tests)

# ------------------------------------------------------ #
print()

# Fitting SVMs
svms = [
    SVC(kernel='linear', C=1).fit(X_train, train_cats),
    SVC(kernel='rbf', C=100, gamma=0.01).fit(X_train, train_cats),
    SVC(kernel='poly', C=1, gamma=0.0001, degree=8).fit(X_train, train_cats)
]

# Calculating scores
for svm in svms:
    kernel = svm.get_params()['kernel']
    score = svm.score(X_tests, tests_cats)
    accuracy = score * 100
    correct = score * total_tests
    print("Accuracy when using %s kernel: %.2f%% (%d out of %d)" % (kernel, accuracy, correct, total_tests))

# ------------------------------------------------------ #
print()

# Parameters for GridSearch
param_grid = [
    {'kernel': ['linear'], 'C': [1, 1e1, 1e2, 1e3]},
    {'kernel': ['rbf'],    'C': [1, 1e1, 1e2, 1e3], 'gamma': [1e-4, 1e-3, 1e-2]},
    {'kernel': ['poly'],   'C': [1, 1e1, 1e2, 1e3], 'gamma': [1e-4, 1e-3, 1e-2], 'degree': [2, 8, 10]},
]

# Perform GridSearch
print("Performing cross-validation...")
cv = GridSearchCV(SVC(), param_grid, cv=10, n_jobs=1).fit(X_train, train_cats)
print("...cross-validation done.")

# Print all GridSearch scores
print("GridSearch scores:")
means = cv.cv_results_['mean_test_score']
stdevs = cv.cv_results_['std_test_score']
params = cv.cv_results_['params']
for mean, stdev, param in zip(means, stdevs, params):
    print("%0.3f (+/-%0.03f) for %r" % (mean, stdev * 2, param))

# Print best parameters for greatest score
print("\nBest parameters: %r" % cv.best_params_)
