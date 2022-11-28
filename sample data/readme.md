sample_data.csv: This is a sample data file containing tweet ids for 172 conversations about DST and 72 conversations about immigration, as described in our paper.

- tweet_id: Unique ID of the tweet
- parent_id: tweet_id of the tweet to which the current tweet is a reply
- label: conversation level tweet annotation as described in section 3.5 of our paper.
- conv_id: unique ID assigned to all tweets belonging to the same conversation (note that this is not the conversation_id returned by Twitter API)
- subject: topic to which the tweet belongs