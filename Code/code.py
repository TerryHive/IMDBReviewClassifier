import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
import numpy as np

# Ζητάμε από τον χρήστη να εισάγει τις διαδρομές
pos_train_path = input("Εισάγετε τη διαδρομή για τα θετικά δεδομένα εκπαίδευσης: ")
neg_train_path = input("Εισάγετε τη διαδρομή για τα αρνητικά δεδομένα εκπαίδευσης: ")
pos_test_path = input("Εισάγετε τη διαδρομή για τα θετικά δεδομένα δοκιμής: ")
neg_test_path = input("Εισάγετε τη διαδρομή για τα αρνητικά δεδομένα δοκιμής: ")

   
# Δηλώνουμε τις διαδρομές των φακέλων με τα θετικά και αρνητικά κείμενα.
p#ositive_folder = "C:\\Users\ΛΕΥΤΕΡΗΣ ΒΕΡΟΥΧΗΣ\\Downloads\\aclImdb_v1\\aclImdb\\train\\pos"
#negative_folder = "C:\\Users\\ΛΕΥΤΕΡΗΣ ΒΕΡΟΥΧΗΣ\\Downloads\\aclImdb_v1\\aclImdb\\train\\neg"
# Η συνάρτηση load_texts_from_folder φορτώνει κείμενα από έναν φάκελο.
def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append(text)
    return texts
# Φόρτωση θετικών και αρνητικών κειμένων από τους αντίστοιχους φακέλους.


   # Φορτώνουμε τα δεδομένα από τις καθορισμένες διαδρομές
positive_train_texts = load_texts_from_folder(pos_train_path)
negativ_train_texts = load_texts_from_folder(neg_train_path)
positive_test_texts = load_texts_from_folder(pos_test_path)
negative_test_texts = load_texts_from_folder(neg_test_path)



#  λεξικό με τις συχνότητες των λέξεων από θετικά και αρνητικά κείμενα

def create_combined_dictionary(positive_texts, negative_texts):
    # Συνδυάζουμε όλα τα κείμενα σε ένα μεγάλο κείμενο.
    all_text = ' '.join(positive_texts + negative_texts)
     # Χρησιμοποιούμε τη βιβλιοθήκη NLTK για να διαχωρίσουμε το κείμενο σε λέξεις (tokenization)
    words = nltk.word_tokenize(all_text)
    words = [word.lower() for word in words if word.isalpha()]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Create a combined dictionary with unique words and their frequencies
    combined_dictionary = {word: count for word, count in word_counts.items()}

    return combined_dictionary

# Δημιουργούμε το λεξικό συχνοτήτων λέξεων από τα θετικά και αρνητικά κείμενα.
combined_dictionary = create_combined_dictionary(positive_train_texts, negative_train_texts)

# Εκτυπώνουμε τον αριθμό των μοναδικών λέξεων στο λεξικό.
#print(len(combined_dictionary))

# Φιλτράρουμε το λεξικό αφαιρώντας τις πιο συχνές και τις λιγότερο συχνές λέξεις.
# Αφαιρούμε τις πρώτες 250 λέξεις και τις τελευταίες 70,300 λέξεις από το λεξικό.
filtered_dict = dict(list(combined_dictionary.items())[250:-70300])

# Εκτυπώνουμε τον αριθμό των λέξεων στο φιλτραρισμένο λεξικό.
#print(len(filtered_dict))

# Δημιουργούμε μια λίστα με τις λέξεις του φιλτραρισμένου λεξικού .
lexicon = filtered_dict.keys()

# Εκτυπώνουμε τον αριθμό των λέξεων στο φιλτραρισμένο λεξικό.
#print(len(list(lexicon)))


# Συνάρτηση για τη δημιουργία λίστας με 0 και 1
def create_binary_list(text, lexicon):
    words = nltk.word_tokenize(text) # χωριζει το κειμενο σε λεξεις 
    words = [word.lower() for word in words if word.isalpha()]
    binary_list = [1 if word in words else 0 for word in lexicon] #1 αν η λεξη ειναι εντος λεξικου, αλλιως 0
    
    return binary_list
X_test=[]
y_test=[]
#X_test=[]    
#X_train = []
#y_train = []
#y_test=[]
for filename in os.listdir(pos_test_path):
    file_path = os.path.join(pos_test_path, filename)

    # Διαβάστε το κείμενο από το αρχείο
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Δημιουργία λίστας με 0 και 1 για το κείμενο
    binary_list = create_binary_list(text, lexicon)

    # Προσθήκη της λίστας στο x_train
    X_test.append(binary_list)

    # Προσθήκη της ετικέτας (θετική κριτική) στο y_train
    y_test.append(1)

for filename in os.listdir(neg_test_path):
    file_path = os.path.join(neg_test_path, filename)

    # Διαβάστε το κείμενο από το αρχείο
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Δημιουργία λίστας με 0 και 1 για το κείμενο
    binary_list = create_binary_list(text, lexicon)

    # Προσθήκη της λίστας στο x_train
    X_test.append(binary_list)

    # Προσθήκη της ετικέτας (αρνητική κριτική) στο y_train
    y_test.append(0)

# Μετατροπή των λιστών σε πίνακες NumPy

X_test = np.array(X_test)
y_test = np.array(y_test)



X_train=[]
y_train=[]
#X_test=[]    
#X_train = []
#y_train = []
#y_test=[]
for filename in os.listdir(pos_train_path):
    file_path = os.path.join(pos_train_path, filename)

    # Διαβάστε το κείμενο από το αρχείο
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Δημιουργία λίστας με 0 και 1 για το κείμενο
    binary_list = create_binary_list(text, lexicon)

    # Προσθήκη της λίστας στο x_train
    X_train.append(binary_list)

    # Προσθήκη της ετικέτας (θετική κριτική) στο y_train
    y_train.append(1)

for filename in os.listdir(neg_train_path):
    file_path = os.path.join(neg_train_path, filename)

    # Διαβάστε το κείμενο από το αρχείο
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Δημιουργία λίστας με 0 και 1 για το κείμενο
    binary_list = create_binary_list(text, lexicon)

    # Προσθήκη της λίστας στο x_train
    X_train.append(binary_list)

    # Προσθήκη της ετικέτας (αρνητική κριτική) στο y_train
    y_train.append(0)

# Μετατροπή των λιστών σε πίνακες NumPy

X_train = np.array(X_train)
y_train = np.array(y_train)







# Εκτύπωση του μεγέθους των x_train και y_train
#print("Μέγεθος X train:", X_train.shape, " ", X_train[200], y_train[200])
#print("Μέγεθος X test :", X_test.shape, " ", X_test[20], y_test[20])
#print("Μέγεθος y:", y_test.shape)












import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities = None
        self.word_probabilities = None


    def evaluate_metrics(self, X_test, y_test):
        predictions = self.predict(X_test)
        true_positives = np.sum((predictions == 1) & (y_test == 1))#υπολογιζεται πληθος αληθινων pos αναλογα με το prediction και την πραγματικη ετικετα
        false_positives = np.sum((predictions == 1) & (y_test == 0))#ομοιως για τα αρνητικα Pos
        false_negatives = np.sum((predictions == 0) & (y_test == 1))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1
    #υπολογιζουμε τις μετρικες αναλογα με τις τιμες που υπολογισαμε παραπανω 

    def fit(self, X_train, y_train):
        num_docs, num_words = X_train.shape
        num_classes = len(np.unique(y_train))

        self.class_probabilities = np.zeros(num_classes) # υπολογισμος πιθανοτητας της καθε κατηγοριας
        self.word_probabilities = np.zeros((num_classes, num_words))#ομοιως για τις λεξεις 

        for c in range(num_classes): #για καθε κατηγορια  (θετικη/αρνητικη)
            class_docs = X_train[y_train == c] #υπολογιζουμε πληθος εγγραφων καθε κατηγοριας
            self.class_probabilities[c] = class_docs.shape[0] / num_docs #πιθανοτητα κατηγοριας
            self.word_probabilities[c] = (class_docs.sum(axis=0) + 1) / (class_docs.sum() + 2) #υπολογισμος πιθανοτητας
            #λεξεων για κατηγορια c  (laplace)

    def predict(self, X_test):
        log_probabilities = np.log(self.word_probabilities)
        log_class_probabilities = np.log(self.class_probabilities)

        predictions = np.argmax(X_test.dot(log_probabilities.T) + log_class_probabilities, axis=1)
        return predictions



# Υποθέτουμε ότι τα X_train, y_train, X_test, y_test είναι διαθέσιμα

#x_test=x_train 
#y_test=y_train
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Υλοποίηση αφελούς ταξινομητή Bayes
#classifier = NaiveBayesClassifier()
#classifier.fit(X_train, y_train)
#predictions = classifier.predict(X_test)

# Υπολογισμός ακρίβειας
#accuracy = np.mean(predictions == y_test)
#print(f"Accuracy: {accuracy}")
#precision, recall, f1 = classifier.evaluate_metrics(X_test, y_test)
#print(f"Precision: {precision}")
#print(f"Recall: {recall}")
#print(f"F1 Score: {f1}")
from sklearn.naive_bayes import GaussianNB

# Δημιουργία του μοντέλου Naive Bayes
#model = GaussianNB()

# Εκπαίδευση του μοντέλου
#model.fit(X_train, y_train)

# Προβλέψεις στο σετ ελέγχου
#y_pred = model.predict(X_test)

# Αξιολόγηση του μοντέλου
#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred, average='macro')
#recall = recall_score(y_test, y_pred, average='macro')
#f1 = f1_score(y_test, y_pred, average='macro')

#print(f"Ακρίβεια (Accuracy): {accuracy}")
#print(f"Ακρίβεια (Precision): {precision}")
#print(f"Ανάκληση (Recall): {recall}")
#print(f"F1 Score: {f1}")

import numpy as np

class TreeNode:
    # Η κλάση TreeNode αναπαριστά έναν κόμβο σε ένα δέντρο αποφάσεων.
    def __init__(self, is_leaf=False, label=None, feature=None, value=None):
        # Αρχικοποιεί έναν κόμβο του δέντρου.
        self.is_leaf = is_leaf  # Εάν ο κόμβος είναι φύλλο (δηλαδή, τελικός κόμβος χωρίς παιδιά).
        self.label = label      # Η ετικέτα που προβλέπει ο κόμβος, χρησιμοποιείται μόνο εάν είναι φύλλο.
        self.feature = feature  # Ο δείκτης τoυ χαρακτηριστικου για τον διαχωρισμό στον κόμβο.
        self.value = value      # Η τιμή του χαρακτηριστικού για τον διαχωρισμό.

        # Αρχικοποιεί τα αριστερά και δεξιά παιδιά του κόμβου ως None.
        # Αυτά θα οριστούν όταν το δέντρο αναπτυχθεί περαιτέρω.
        self.left = None   # Ο αριστερός παιδί κόμβος.
        self.right = None  # Ο δεξιός παιδί κόμβος.

def calculate_entropy(labels):
    # Υπολογίζουμε τη συχνότητα κάθε μοναδικής ετικέτας στο σύνολο ετικετών.
    _, counts = np.unique(labels, return_counts=True) #αριθμός των φορών που εμφανίζεται κάθε μοναδική τιμή πχ για [1, 0, 1, 1, 0]
                                                      #θα ειναι [2, 3]
    # Μετατρέπουμε τις συχνότητες σε πιθανότητες διαιρώντας με το συνολικό αριθμό των ετικετών.
    probabilities = counts / counts.sum()
    
    # Υπολογίζουμε την εντροπία ως το άθροισμα των πιθανοτήτων επί το λογάριθμο των πιθανοτήτων.
    # Χρησιμοποιούμε τον λογάριθμο με βάση 2, καθώς η εντροπία μετράται σε bits.
    return -np.sum(probabilities * np.log2(probabilities))

def split_data(data, labels, feature, value):
    # Δημιουργούμε μια μάσκα για τον διαχωρισμό των δεδομένων.
    # Η left_mask θα έχει True στις θέσεις όπου η τιμή του χαρακτηριστικού είναι μικρότερη ή ίση με την δοθείσα τιμή.
    left_mask = data[:, feature] == 0 #πίνακας από τιμές True και False που δείχνει ποιες γραμμές του data πληρούν το κριτήριο.

    # Αντίστοιχα, η right_mask θα έχει True όπου η τιμή είναι μεγαλύτερη από την δοθείσα τιμή.
    right_mask = data[:, feature] ==1

    # Επιστρέφουμε τα υποσύνολα των δεδομένων και των ετικετών βάσει των μασκών.
    # Τα δεδομένα και οι ετικέτες που αντιστοιχούν στη left_mask αποτελούν το "αριστερό" υποσύνολο,
    # ενώ αυτά που αντιστοιχούν στη right_mask αποτελούν το "δεξιό" υποσύνολο.
    return data[left_mask], labels[left_mask], data[right_mask], labels[right_mask]  #γυρναει τις αντιστοιχες γραμμες 


def find_best_split(data, labels):
    # Αρχικοποιεί το καλύτερο κέρδος πληροφορίας και τον καλύτερο διαχωρισμό.
    best_gain = -1
    best_split = None

    # Υπολογίζει την τρέχουσα εντροπία του συνόλου των δεδομένων.
    current_entropy = calculate_entropy(labels)
    
    # Επαναληπτικά εξετάζει κάθε χαρακτηριστικό των δεδομένων.
    for feature in range(data.shape[1]):  #range=στηλες δηλαδη καθε χαρακτηριστικο
        # Αποκτά όλες τις μοναδικές τιμές για το τρέχον χαρακτηριστικό.
        values = np.unique(data[:, feature])

        # Εξετάζει κάθε μοναδική τιμή ως πιθανό σημείο διαχωρισμού.
        for value in values:
            # Διαχωρίζει τα δεδομένα βάσει του τρέχοντος χαρακτηριστικού και τιμής.
            left_data, left_labels, right_data, right_labels = split_data(data, labels, feature, value)

            # Ελέγχει αν ο διαχωρισμός είναι έγκυρος (και τα δύο υποσύνολα πρέπει να έχουν δεδομένα).
            if len(left_data) == 0 or len(right_data) == 0:
                continue
            
            # Υπολογίζει την εντροπία για κάθε υποσύνολο.
            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)

            # Υπολογίζει το κέρδος πληροφορίας για αυτόν τον διαχωρισμό.
            gain = current_entropy - (left_entropy * len(left_labels) + right_entropy * len(right_labels)) / len(labels)
            
            # Ελέγχει αν αυτός ο διαχωρισμός παρέχει το καλύτερο κέρδος πληροφορίας μέχρι στιγμής.
            if gain > best_gain:
                best_gain = gain
                best_split = (feature, value)
    
    # Επιστρέφει τον καλύτερο διαχωρισμό που βρέθηκε.
    return best_split
    
#Αυτή η συνάρτηση κατασκευάζει ένα δέντρο αποφάσεων.
   # Χρησιμοποιεί τ αναδρομή για να φτιαξει υποδέντρα
def build_tree(data, labels, depth):
    # Ελέγχουμε μόνο αν όλες οι ετικέτες είναι ίδιες 
    if len(np.unique(labels)) == 1:
        return TreeNode(is_leaf=True, label=labels[0]) # δημιουργει φυλλο, δεν χρειαζεται παραπανω διαχωρισμος

    split = find_best_split(data, labels)
    if split is None:
        label = np.argmax(np.bincount(labels)) #bicount taksinomei to plithos twn 0 kai 1 , argmax ton deikti tis megistis timis ->epistrefei tin pio sixni etiketa
        return TreeNode(is_leaf=True, label=label) # den mporei na diaxoristei parapano , den iparxei information gain ara einai leaf
    
    feature, value = split  # apothikefsi apotelesmaton tou  best split
    left_data, left_labels, right_data, right_labels = split_data(data, labels, feature, value) # ta 
    node = TreeNode(feature=feature, value=value)
    node.left = build_tree(left_data, left_labels, depth + 1)
    node.right = build_tree(right_data, right_labels, depth + 1)
    
    return node
 
 #υλοποιεί τον αλγόριθμο Random Forest 

def random_forest(data, labels, num_trees):
    trees = []
    for _ in range(num_trees):
        indices = np.random.choice(len(data), len(data)) #παιρνει τυχαια δειγματα και φτιαχνει ενα πινακα size len(data)
        sample_data, sample_labels = data[indices], labels[indices] # παιρνουν τις αντιστοιχες τιμες 
        tree = build_tree(sample_data, sample_labels, 0)
        trees.append(tree)  #φτιαχνει το δασος
    return trees
 # προβλέπει την ετικέτα με  δέντρο αποφάσεων.
    #Η διαδικασία είναι αναδρομική και ακολουθεί τα κριτήρια διαχωρισμού του δέντρου μέχρι να φτάσει σε ένα φύλλο.
def predict(tree, sample):
    if tree.is_leaf:
        return tree.label
    if (sample[tree.feature] ==0):
        return predict(tree.left, sample)
    return predict(tree.right, sample)

 #Αυτή η συνάρτηση προβλέπει τις ετικέτες για ένα σύνολο δειγμάτων χρησιμοποιώντας τα δέντρα ενός τυχαίου δάσους.
# Για κάθε δείγμα, πραγματοποιεί προβλέψεις από όλα τα δέντρα και επιλέγει την πιο συχνή ετικέτα ως τελική πρόβλεψη
def random_forest_predict(trees, samples):
    predictions = []
    for sample in samples:
        tree_preds = [predict(tree, sample) for tree in trees]
        predictions.append(np.argmax(np.bincount(tree_preds))) #προσθέτει το επικρατεστερο 
    return predictions

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# Scikit-learn Random Forest
#sklearn_forest = RandomForestClassifier(n_estimators=1, random_state=42)
#sklearn_forest.fit(X_train, y_train)
 # εκπαιδευει τον αλγόριθμο Random Forest που φτιαξαμε 
#custom_forest = random_forest(X_train, y_train, num_trees=1)
def plot_learning_curves_and_print_metrics(X_train_og, y_train_og, X_test, y_test, train_sizes, algorithm, num_trees=2):
    metrics_custom_test = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    metrics_sklearn_test = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for train_size in train_sizes:
        size = int(len(X_train_og) * train_size)
        print ("megethos: "+str(size))
        X_train = X_train_og[:size]
        y_train = y_train_og[:size]

        if algorithm == 'Naive Bayes':
            # Προσαρμοσμένος Naive Bayes
            classifier = NaiveBayesClassifier()
            classifier.fit(X_train, y_train)
            y_test_pred_custom = classifier.predict(X_test)

            # Scikit-learn Naive Bayes
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_test_pred_sklearn = model.predict(X_test)
        elif algorithm == 'Random Forest':
            # Προσαρμοσμένος Random Forest
            custom_forest = random_forest(X_train, y_train, num_trees)
            y_test_pred_custom = random_forest_predict(custom_forest, X_test)

            # Scikit-learn Random Forest
            sklearn_forest = RandomForestClassifier(n_estimators=num_trees, random_state=42)
            sklearn_forest.fit(X_train, y_train)
            y_test_pred_sklearn = sklearn_forest.predict(X_test)

        # Υπολογισμός μετρικών για το test set
        for metric in metrics_custom_test.keys():
            func = globals()[f"{metric}_score"]
            metrics_custom_test[metric].append(func(y_test, y_test_pred_custom))
            metrics_sklearn_test[metric].append(func(y_test, y_test_pred_sklearn))

    # Εκτύπωση συγκρίσεων μετρικών για το test set
    for metric in metrics_custom_test.keys():
        print(f"\n{metric} (Test set):")
        for i in range(len(train_sizes)):
            print(f"Train size: {train_sizes[i]}, Custom Test {metric}: {metrics_custom_test[metric][i]:.2f}, "
                  f"Scikit-learn Test {metric}: {metrics_sklearn_test[metric][i]:.2f}")

    # Σχεδίαση καμπυλών μάθησης για το test set
    plt.figure(figsize=(12, 8))
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, metric in enumerate(metrics_custom_test):
        plt.subplot(2, 2, i + 1)
        plt.plot(train_sizes, metrics_custom_test[metric], label='Custom Test')
        plt.plot(train_sizes, metrics_sklearn_test[metric], label='Scikit-learn Test')
        plt.title(f'Learning Curve for {titles[i]} (Test set)')
        plt.xlabel('Training Size')
        plt.ylabel(titles[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

plot_learning_curves_and_print_metrics(X_train, y_train,X_test,y_test, train_sizes,'Naive Bayes')
plot_learning_curves_and_print_metrics(X_train, y_train,X_test,y_test, train_sizes,'Random Forest')

print ("\n αυτο ειναι ενα τεστ γιολο")    