import config
import torch
import numpy as np
import pandas as pd




class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN



    def __len__(self):
        return len(self.tweet)


    #most important function
    def __getitem__(self, item):
	    #get common_text
        tweet = str(self.tweet[item])
        #split common_text string into tokens
        tweet = " ".join(tweet.split())


        selected_text = str(self.selected_text[item])
        selected_text = " ".join(selected_text.split())



        len_sel_text = len(selected_text)


        #staring index in tweet which contains same character in selected text at this index 
        idx0 = -1
        #last index in tweet which contains same character in selected text at this index 
        idx1 = -1
        #from idex0 to idx1 in tweet is equal to the selected text



        #i is index of character of tweet , e is character of tweet 
        #we go through each character of tweet and if tweet's character is same as first character of selected_text then I am interested in that index
        for ind in (i for i,e in enumerate(tweet) if e == selected_text[0]):
            #checking if from starting index to len(selected_text) is equal that means they are same string
            if tweet[ind: ind+len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break



        #tweet is  a string  
        char_targets = [0] * len(tweet)
        #if selected exists in tweet  
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                #if space_character has been found at some index  then set that index as 0 in char_targets array 
                #else set to 1
                if tweet[j] != " ":
                    char_targets[j] = 1



	
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[1:-1]



        # print(tok_tweet_tokens)
        #['[CLS]', 'hello', ',', 'y', "'", 'all', '!', 'how', 'are', 'you', '[UNK]', '?', '[SEP]']


        # print(tok_tweet_offsets)  #list of tuples and tuples are offsets==>> (offset1,offset2)
        #[(0, 0), (0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), (14, 17), (18, 21), (22, 25), (26, 27),(28, 29), (0, 0)]


        #toke_tweet_ids
        #[101, 7592, 1010, 1061, 1005, 2035, 999, 2129, 2024, 2017, 100, 1029, 102]


        #-2 for removing cls and sep tokens
        targets = [0] * (len(tok_tweet_tokens) - 2)


        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            #some characters from one word towards the end they are included ,some character from beginning of a word are included that means if there is a partial match then set target as 1
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] = 1 
        #currently targets' any index's value is 1 if that index represents the word that is present in the selected text



        #targets' shape ==  length of tweet + 2  => here 2 is for cls,sep tokens
        targets = [0] + targets + [0] # add two zeros for cls and sep because previously we removed it
        targets_start = [0] * len(targets)
        targets_end  =  [0] * len(targets)



        #get indices of non-zero values present in targets array
        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            #set starting word's value of selected text in target_start array =1
            targets_start[non_zero[0]] = 1
            #set last word's value of selected text in target_end array =1
            targets_end[non_zero[-1]] = 1
        



        #attention mask
        mask  = [1] * len(tok_tweet_ids)
        token_type_ids  = [0] * len(tok_tweet_ids)



        #padding is done with zeros
        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)



        #encode sentiment columns
        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]





        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            
            'tweet_tokens': " ".join(tok_tweet_tokens),
            
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }




if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE.dropna().reset_index(drop=True))
    dset = TweetDataset(
            tweet = df.text.values,
            sentiment = df.sentiment.values,
            selected_text = df.selected_text.values

        )

    #print first 
    print(dset[0])  #it is a dict of above return value

