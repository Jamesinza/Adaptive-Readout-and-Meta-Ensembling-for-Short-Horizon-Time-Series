import numpy as np
import pandas as pd
import keras
import random
from tensorflow import random as tf_random
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf_random.set_seed(SEED)

def build_model(INPUT_SHAPE, DIM, SEED, NUM_CLASSES):
    inputs = keras.layers.Input(shape=INPUT_SHAPE)
    x=norm(inputs)

    x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED)(x)
    # x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED+1)(x1)
    # x1 = keras.layers.Dropout(0.3)(x1)
    
    x2 = keras.layers.LSTM(DIM, return_sequences=True, seed=SEED)(x)
    # x2 = keras.layers.Dropout(0.3)(x2)
    
    x3 = keras.layers.Conv1D(DIM, 3, padding='same')(x)
    x3 = keras.layers.Activation('gelu')(x3)
    # x3 = keras.layers.Dropout(0.3)(x3)

    x4 = keras.layers.Dense(DIM, activation='gelu')(x)
    
    conc     = [x1,x2,x3,x4]
    fusion   = keras.layers.Concatenate()(conc)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(fusion, fusion)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    # weights  = keras.layers.Dense(len(conc), activation='softmax')(fusion)

    # w1 = keras.layers.Lambda(lambda w: w[:, :, 0:1])(weights)
    # w2 = keras.layers.Lambda(lambda w: w[:, :, 1:2])(weights)
    # w3 = keras.layers.Lambda(lambda w: w[:, :, 2:3])(weights)
    # x = keras.layers.Add()([
    #     keras.layers.Multiply()([x1, w1]),
    #     keras.layers.Multiply()([x2, w2]),
    #     keras.layers.Multiply()([x3, w3]),
    # ])

    # x = keras.layers.TimeDistributed(keras.layers.Dense(DIM, activation='gelu'))(x)
    # x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

def calculate_streaks(pred, y_test):
    losses = 0
    wins = 0
    losing_streak = 0
    win_streak = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            continue
        elif (pred[i] == 1 or pred[i] == 2) and (y_test[i] == 1 or y_test[i] == 2):
            wins += 1
            win_streak = max(wins, win_streak)
            losses = 0
        else:
            wins = 0
            losses += 1
            losing_streak = max(losses, losing_streak)
    return losing_streak, win_streak

def create_sequences(data, label, WL, TARGET):
    X_data = np.empty([len(data) - WL, WL, data.shape[-1]], dtype=np.float32)
    y_data = np.empty([len(data) - WL], dtype=np.int8)
    for i in range(len(data) - WL):
        X_data[i] = data[i:i+WL]
        y_data[i] = label[i+WL, TARGET]
    return X_data, y_data

def create_features(app_data, WL, TARGET):
    # base features
    app_data['pct'] = app_data['Payout'].pct_change()
    app_data['skew'] = app_data['pct'].rolling(WL).skew()
    app_data['kurt'] = app_data['pct'].rolling(WL).kurt()
    # aggregate features
    app_data['pct_mean'] = app_data['pct'].rolling(WL).mean()
    app_data['skew_mean'] = app_data['skew'].rolling(WL).mean()
    app_data['kurt_mean'] = app_data['kurt'].rolling(WL).mean()
    # correlations
    app_data['corr1'] = app_data['pct'].rolling(WL).corr(app_data['skew'])
    app_data['corr2'] = app_data['pct'].rolling(WL).corr(app_data['kurt'])
    app_data['corr3'] = app_data['skew'].rolling(WL).corr(app_data['kurt'])
    app_data['corr4'] = app_data['pct_mean'].rolling(WL).corr(app_data['skew_mean'])
    app_data['corr5'] = app_data['pct_mean'].rolling(WL).corr(app_data['kurt_mean'])
    app_data['corr6'] = app_data['skew_mean'].rolling(WL).corr(app_data['kurt_mean'])
    # combined features
    app_data.dropna(inplace=True)
    app_data.reset_index(inplace=True, drop=True)
    data = app_data[['pct','skew','kurt','pct_mean','skew_mean','kurt_mean',
                     'corr1','corr2','corr3','corr4','corr5','corr6',
                    ]].copy()
    data = data.values
    label = app_data[['label']].values
    X_data, y_data = create_sequences(data, label, WL, TARGET)
    return X_data, y_data

df = pd.read_csv('datasets/aviator_dataset.csv')

if 'DateTime' not in df.columns:
    if 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
    else:
        raise RuntimeError("No DateTime or Date+Time columns found in CSV.")
else:
    df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df[['DateTime','App','Payout']]

df = df.sort_values('DateTime').reset_index(drop=True)
# df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute
# df = df[df['Payout'] >= 20.0]

df = df[df['month'] == 5]

df['label'] = np.where(
    df['Payout'] < 2, 0, np.where(df['Payout'] < 10, 1, 2)
)

apps = ['WINPESA', 'ODIBETS'] # ,'BETIKA','BETGR8','ODIBETS']

WL = 64
DIM = 1
TARGET = 0
models = []

for APP in apps:
    keras.backend.clear_session()
    
    train_data = df[df['App'] == APP].copy()
    X_train, y_train = create_features(train_data, WL, TARGET)

    # test_data1 = df[df['App'] == 'ODIBETS'].copy()
    # X_test1, y_test1 = create_features(test_data1, WL, TARGET)

    # test_data2 = df[df['App'] == 'BETGR8'].copy()
    # X_test2, y_test2 = create_features(test_data2, WL, TARGET)

    # test_data3 = df[df['App'] == 'BETIKA'].copy()
    # X_test3, y_test3 = create_features(test_data3, WL, TARGET)
    
    print(f'\nX_train: {X_train.shape}')
    print(f'\ny_train: {y_train.shape}\n')

    unique_classes     = np.unique(y_train.flatten())
    class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    NUM_CLASSES        = len(unique_classes)    

    norm = keras.layers.Normalization()
    norm.adapt(X_train)

    INPUT_SHAPE = (WL, X_train.shape[-1])
    
    # inputs = keras.layers.Input(shape=INPUT_SHAPE)
    # x=norm(inputs)

    # x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED)(x)
    # # x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED+1)(x1)
    # # x1 = keras.layers.Dropout(0.3)(x1)
    
    # x2 = keras.layers.LSTM(DIM, return_sequences=True, seed=SEED)(x)
    # # x2 = keras.layers.Dropout(0.3)(x2)
    
    # x3 = keras.layers.Conv1D(DIM, 3, padding='same')(x)
    # x3 = keras.layers.Activation('gelu')(x3)
    # # x3 = keras.layers.Dropout(0.3)(x3)

    # x4 = keras.layers.Dense(DIM, activation='gelu')(x)
    
    # conc     = [x1,x2,x3,x4]
    # fusion   = keras.layers.Concatenate()(conc)
    # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(fusion, fusion)
    # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    # # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    # # weights  = keras.layers.Dense(len(conc), activation='softmax')(fusion)

    # # w1 = keras.layers.Lambda(lambda w: w[:, :, 0:1])(weights)
    # # w2 = keras.layers.Lambda(lambda w: w[:, :, 1:2])(weights)
    # # w3 = keras.layers.Lambda(lambda w: w[:, :, 2:3])(weights)
    # # x = keras.layers.Add()([
    # #     keras.layers.Multiply()([x1, w1]),
    # #     keras.layers.Multiply()([x2, w2]),
    # #     keras.layers.Multiply()([x3, w3]),
    # # ])

    # # x = keras.layers.TimeDistributed(keras.layers.Dense(DIM, activation='gelu'))(x)
    # # x = keras.layers.Dropout(0.3)(x)
    # x = keras.layers.Flatten()(x)
    # outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    # model = keras.Model(inputs, outputs)
    model = build_model(INPUT_SHAPE, DIM, SEED, NUM_CLASSES)
    # model.summary()

    model.compile(optimizer=keras.optimizers.AdamW(1e-3), loss='sparse_categorical_crossentropy')
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, cooldown=0),
    ]

    model.fit(X_train, y_train, validation_split=0.1, verbose=1, epochs=2,
               batch_size=128, callbacks=callbacks, class_weight=class_weights_dict,
               shuffle=False,
                       )
    model.save(f'models/{APP}_{TARGET}.keras')
    models.append(model)

    # preds1 = model.predict(X_test1)
    # pred1 = np.argmax(preds1, axis=1)

# for model in models:
test_data2 = df[df['App'] == 'BETGR8'].copy()
X_test2, y_test2 = create_features(test_data2, WL, TARGET)

test_data3 = df[df['App'] == 'BETIKA'].copy()
X_test3, y_test3 = create_features(test_data3, WL, TARGET)

preds2 = models[0].predict(X_test2)
pred2 = np.argmax(preds2, axis=1)

preds3 = models[1].predict(X_test2)
pred3 = np.argmax(preds3, axis=1)


# # print(f'\nAccuracy on Test1 set: {np.mean(pred1==y_test1):.2f}')
# print(f'\nAccuracy on Test2 set: {np.mean(pred2==y_test2):.2f}')
# print(f'\nAccuracy on Test3 set: {np.mean(pred3==y_test3):.2f}')

# # losing_streak1, win_streak1 = calculate_streaks(pred1, y_test1)
# losing_streak2, win_streak2 = calculate_streaks(pred2, y_test2)
# losing_streak3, win_streak3 = calculate_streaks(pred3, y_test3)

# # print('\nSample Results')
# # print(pred[:50])
# # print(y_test[:50])

# # print(f'\nLongest losing streak for Test1  = {losing_streak1}')
# # print(f'Longest winning streak for Test1 = {win_streak1}')

# print(f'\nLongest losing streak for Test2  = {losing_streak2}')
# print(f'Longest winning streak for Test2 = {win_streak2}')

# print(f'\nLongest losing streak for Test3  = {losing_streak3}')
# print(f'Longest winning streak for Test3 = {win_streak3}')
