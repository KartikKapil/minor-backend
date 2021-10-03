import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

# ## Task 3: Importing Data
# 
# 1. Importing the Tweet Emotion dataset
# 2. Creating train, validation and test sets
# 3. Extracting tweets and labels from the examples

def get_data():
    dataset = nlp.load_dataset('emotion')
    return dataset


def get_sequences(tokenizer,tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences,truncating='post',padding='post',maxlen=50)
    return padded

def get_tweet(data):
    tweets = [x['text'] for x in data]
    label = [x['label'] for x in data]
    return tweets,label

def preprocessing_data():
    dataset = get_data()
    train = dataset['train']
    val = dataset['validation']
    test = dataset['test']

    tweets , labels = get_tweet(train)
    # ## Task 4: Tokenizer 
    # 1. Tokenizing the tweets
    tokenizer = Tokenizer(num_words=10000,oov_token='<UNK>')
    tokenizer.fit_on_texts(tweets)
    tokenizer.texts_to_sequences([tweets[0]])
    # this is use to check how it has converted words to numbers ie done encoding 
    # tokenizer.word_index <-- to check the dict of words tokenized

    # ## Task 5: Padding and Truncating Sequences
    # 
    # 1. Checking length of the tweets
    # 2. Creating padded sequences

    lengths = [len(t.split(' ')) for t in tweets]
    # plt.hist(lengths,bins = len(set(lengths))) # the bins are set to the unique length of the 'lengths' array
    # plt.show()

    maxlen = 50 # if the length is more the 50 we are going to short them, else pad them 

    padded_train_seq = get_sequences(tokenizer,tweets)

    # ## Task 6: Preparing the Labels
    # 
    # 1. Creating classes to index and index to classes dictionaries
    # 2. Converting text labels to numeric labels
    classes = set(labels) # total number of different classes we have 

    # plt.hist(labels,bins=11)
    # plt.show() #  we have these classes and the classes are a little imblance

    class_to_index = dict((c,i) for i,c in enumerate(classes))
    index_to_class = dict((v,k) for k,v in class_to_index.items())

    names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels]) 

    train_labels = names_to_ids(labels)
    # print(train_labels[0])


    # ## Task 7: Creating the Model
    # 
    # 1. Creating the model
    # 2. Compiling the model


    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000,16,input_length=maxlen),# Each word will essential make into a 16 vector(vocab_Size,dimensions of vector) 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6,activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


    model.summary()


    # ## Task 8: Training the Model
    # 
    # 1. Preparing a validation set
    # 2. Training the model

    val_tweets,val_labels = get_tweet(val)
    val_seq = get_sequences(tokenizer,val_tweets)
    val_labels = names_to_ids(val_labels)


    val_tweets[0],val_seq[0],val_labels[0]


    h = model.fit(
        padded_train_seq, train_labels,
        validation_data=(val_seq,val_labels),
        epochs=25,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2)
        ]
    )


    # ## Task 9: Evaluating the Model
    # 
    # 1. Visualizing training history
    # 2. Prepraring a test set
    # 3. A look at individual predictions on the test set
    # 4. A look at all predictions on the test set

    # In[54]:


    # show_history(h)

    model.save('main_model.h5')


def test_model():
    """DO NOT RUN"""
    test_tweets,test_labels = get_tweet(test)
    test_seq = get_sequences(tokenizer,test_tweets)
    test_labels = names_to_ids(test_labels)


    _ = model.evaluate(test_seq,test_labels)

    i = random.randint(0,len(test_labels)-1)

    print('sentence:',test_tweets[i])
    print('emotion:',index_to_class[test_labels[i]])

    p = model.predict(np.expand_dims(test_seq[i],axis=0))[0]
    pred_class = index_to_class[np.argmax(p).astype('uint8')]

    print('predicted class',pred_class)


    preds  = model.predict(test_seq)
    classes_x=np.argmax(preds,axis=1)



# show_confusion_matrix(test_labels,classes_x,list(classes))

def incoming_message(message):
    model = tf.keras.models.load_model('main_model.h5')
    a=[]
    a.append(message)
    tokenizer = Tokenizer(num_words=10000,oov_token='<UNK>')
    t=get_sequences(tokenizer,a)

    p = model.predict(np.expand_dims(t[0],axis=0))[0]
    # pred_class = index_to_class[np.argmax(p).astype('uint8')]

    print('predicted class',p)


get_data()
preprocessing_data()
incoming_message("i am happy")