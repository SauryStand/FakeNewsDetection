

def clean_data(dataframe):
    # Drop duplicate rows
    dataframe.drop_duplicates(subset='tweetText', inplace=True)

    # Remove punctation
    dataframe['tweetText'] = dataframe['tweetText'].str.replace('[^\w\s]', ' ')

    # Remove numbers
    dataframe['tweetText'] = dataframe['tweetText'].str.replace('[^A-Za-z]', ' ')

    # Make sure any double-spaces are single
    dataframe['tweetText'] = dataframe['tweetText'].str.replace('  ', ' ')
    dataframe['tweetText'] = dataframe['tweetText'].str.replace('  ', ' ')

    # Transform all text to lowercase
    dataframe['tweetText'] = dataframe['tweetText'].str.lower()

    print("New shape:", dataframe.shape)
    return dataframe.head()