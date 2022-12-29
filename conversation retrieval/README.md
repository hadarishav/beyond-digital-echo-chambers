**FILES AND INSTRUCTIONS**

---

1_retrieve_conversation_ids.py: This file retrieves conversation id per tweet id. Needs an excel file with tweet_ids as a column as input, and config file with Twitter API tokens.

```jsx
python 1_retrieve_conversation_ids.py --file_name "input.xlsx"
```

2_retrieve_conversations.py: This file takes conversation_ids retrieved from previous step as input and retrieve conversations using Twitter API.

```jsx
python 2_retrieve_conversations.py
```

3_conversation_reconstruction.py: This file takes retrieved conversations from previous code as input and reconstructs a conversation tree per unique conversation_id.

```jsx
python 3_conversation_reconstruction.py
```
