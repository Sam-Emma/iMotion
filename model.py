def predict_emotion(text):
    import pandas as pd
    import re
    data = pd.read_csv('emotionText.csv')  
    print(data)
    test = pd.DataFrame()
    test['text'] = [text]
    
    def split_it(tt):
        return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tt)
    data['text'] = data['text'].apply(lambda x: split_it(x))
    data['text'] = data['text'].apply(lambda x: x.replace('RT',''))
    test['text'] = test['text'].apply(lambda x: split_it(x))
    test['text'] = test['text'].apply(lambda x: x.replace('RT',''))
 
    data['question'] = data['text'].apply(lambda x: len([x for x in x.split() if x.endswith('?')]))
    data['text'] = data['text'].apply(lambda x: x.lower())
    
    
    test['question'] = test['text'].apply(lambda x: len([x for x in x.split() if x.endswith('?')]))
    test['text'] = test['text'].apply(lambda x: x.lower())
    test.head(5)
    
    data['text'] = data['text'].str.replace('[^\w\s]','')
    
    test['text'] = test['text'].str.replace('[^\w\s]','')
    #from textblob import TextBlob
#    data['text'] = data['text'].apply(lambda x: str(TextBlob(x).correct()))
#    
#    test['text'] = test['text'].apply(lambda x: str(TextBlob(x).correct()))
    
   
    
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=3)
    data_text = vectorizer.fit_transform(data['text'])
    test_text = vectorizer.transform(test['text'])
   
    df = pd.DataFrame(data_text.toarray())
    df_t = pd.DataFrame(test_text.toarray())
    mergedata = data.merge(df, left_index=True, right_index=True)
    mergetestdata = test.merge(df_t, left_index=True, right_index=True)
    data = mergedata
    test_pre = mergetestdata
   
    
    
    X = data.drop(['ID','text','target'],axis = 1)
    test_pre = test_pre.drop(['text'],axis = 1)
    y = data['target']
    
#    from sklearn.model_selection import train_test_split
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#    
#    print(X_train)
#    print(text)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X, y)
    import numpy as np
    # Make predictions
    confidence = 0
    predict_real = ""
    prediction = lr.predict(test_pre)
    prob_prediction = lr.predict_proba(test_pre)
    if (prob_prediction[0][1] < 50 and prediction[0] == 0):
        confidence = str(np.round(prob_prediction[0][0],4)*100)+"%"
    else:
        confidence = str(np.round(prob_prediction[0][1],4)*100)+"%"
    if(prediction[0] == 0):
        predict_real = "Unhappy"
    else:
        predict_real = "Happy"
    return text, predict_real, confidence
#text,pred,con = predict_emotion("Congratulations")
#print(text,pred,con)
    
