
# GRU
model = Sequential()
model.add(Embedding(train_words_count, 128, input_length=X_train2.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# BiLSTM
model = Sequential()
model.add(Embedding(train_words_count, 128, input_length=X_train2.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# GRU
model = Sequential()
model.add(Embedding(train_words_count, 128, input_length=X_train2.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# BiGRU
model = Sequential()
model.add(Embedding(train_words_count, 128, input_length=X_train2.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])