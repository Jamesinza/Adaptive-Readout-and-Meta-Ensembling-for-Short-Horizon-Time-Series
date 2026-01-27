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

@keras.utils.register_keras_serializable(package="Custom")
class LearnableQueryPooling(keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query = self.add_weight(
            shape=(1, dim),
            initializer="glorot_uniform",
            trainable=True,
            name="learnable_query"
        )

    def call(self, x):
        # x: (B, T, D)
        q = self.query  # (1, D)
        q = tf.expand_dims(q, axis=1)  # (1, 1, D)
        # Compute attention scores
        scores = tf.matmul(q, x, transpose_b=True)  # (1, 1, T)
        scores = tf.nn.softmax(scores, axis=-1)     # (1, 1, T)
        # Weighted sum
        context = tf.matmul(scores, x)              # (1, 1, D)
        context = tf.squeeze(context, axis=1)       # (1, D)
        return context

def build_model(norm, INPUT_SHAPE, DIM, SEED, NUM_CLASSES, LAYERS):
    inputs = keras.layers.Input(shape=INPUT_SHAPE)
    x=norm(inputs)
    for i in range(LAYERS):
        x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED+i)(x)
        # x1 = keras.layers.GRU(DIM, return_sequences=True, seed=SEED+1)(x1)
        x1 = keras.layers.Dropout(0.3)(x1)
        
        x2 = keras.layers.LSTM(DIM, return_sequences=True, seed=SEED+i)(x)
        x2 = keras.layers.Dropout(0.3)(x2)
        
        x3 = keras.layers.Conv1D(DIM, 3, padding='same')(x)
        x3 = keras.layers.Activation('gelu')(x3)
        x3 = keras.layers.Dropout(0.3)(x3)
    
        x4 = keras.layers.TimeDistributed(keras.layers.Dense(DIM, activation='gelu'))(x)
        x4 = keras.layers.Dropout(0.3)(x4)
        
        conc    = [x1,x2,x3,x4]
        fusion  = keras.layers.Concatenate()(conc)
        # fusion  = keras.layers.Dropout(0.3)(fusion)
    
        # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(fusion, fusion)
        # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
        # x = keras.layers.MultiHeadAttention(num_heads=4, key_dim=4)(x,x)
    
        weights  = keras.layers.Dense(len(conc), activation='softmax')(fusion)
    
        w1 = keras.layers.Lambda(lambda w: w[:, :, 0:1])(weights)
        w2 = keras.layers.Lambda(lambda w: w[:, :, 1:2])(weights)
        w3 = keras.layers.Lambda(lambda w: w[:, :, 2:3])(weights)
        w4 = keras.layers.Lambda(lambda w: w[:, :, 3:4])(weights)
        
        x = keras.layers.Add()([
            keras.layers.Multiply()([x1, w1]),
            keras.layers.Multiply()([x2, w2]),
            keras.layers.Multiply()([x3, w3]),
            keras.layers.Multiply()([x4, w4]),
        ])

    x = keras.layers.TimeDistributed(keras.layers.Dense(DIM, activation='gelu'))(x)
    x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.LSTM(DIM)(x)
    x = LearnableQueryPooling(DIM)
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
    # # base features
    # app_data['pct'] = app_data['Payout'].pct_change()
    # app_data['skew'] = app_data['pct'].rolling(WL).skew()
    # app_data['kurt'] = app_data['pct'].rolling(WL).kurt()
    # # aggregate features
    # app_data['pct_mean'] = app_data['pct'].rolling(WL).mean()
    # app_data['skew_mean'] = app_data['skew'].rolling(WL).mean()
    # app_data['kurt_mean'] = app_data['kurt'].rolling(WL).mean()
    # # correlations
    # app_data['corr1'] = app_data['pct'].rolling(WL).corr(app_data['skew'])
    # app_data['corr2'] = app_data['pct'].rolling(WL).corr(app_data['kurt'])
    # app_data['corr3'] = app_data['skew'].rolling(WL).corr(app_data['kurt'])
    # app_data['corr4'] = app_data['pct_mean'].rolling(WL).corr(app_data['skew_mean'])
    # app_data['corr5'] = app_data['pct_mean'].rolling(WL).corr(app_data['kurt_mean'])
    # app_data['corr6'] = app_data['skew_mean'].rolling(WL).corr(app_data['kurt_mean'])
    # # combined features
    app_data.dropna(inplace=True)
    app_data.reset_index(inplace=True, drop=True)
    data = app_data[['Payout', # 'pct', 'skew','kurt','pct_mean','skew_mean','kurt_mean',
                     # 'corr1','corr2','corr3','corr4','corr5','corr6',
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

WL = 3
DIM = 18
LAYERS = 2
TARGET = 0
EPOCHS = 1000
models = []

for APP in apps:
    keras.backend.clear_session()
    
    train_data = df[df['App'] == APP].copy()
    X_train, y_train = create_features(train_data, WL, TARGET)
    
    print(f'\nX_train: {X_train.shape}')
    print(f'\ny_train: {y_train.shape}\n')

    unique_classes     = np.unique(y_train.flatten())
    class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y_train.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    NUM_CLASSES        = len(unique_classes)

    norm = keras.layers.Normalization()
    norm.adapt(X_train)

    INPUT_SHAPE = (WL, X_train.shape[-1])
    
    model = build_model(norm, INPUT_SHAPE, DIM, SEED, NUM_CLASSES, LAYERS)
    model.summary()

    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, cooldown=0, min_lr=1e-8),
    ]

    model.fit(X_train, y_train, validation_split=0.1, verbose=1, epochs=EPOCHS,
               batch_size=128, callbacks=callbacks, class_weight=class_weights_dict,
               shuffle=False,
                       )
    model.save(f'models/{APP}_{TARGET}.keras')
    models.append(model)

test_data1 = df[df['App'] == 'WINPESA'].copy()
X_test1, y_test1 = create_features(test_data1, WL, TARGET)

test_data2 = df[df['App'] == 'ODIBETS'].copy()
X_test2, y_test2 = create_features(test_data2, WL, TARGET)

X_test = np.vstack([X_test1, X_test2])
y_test = np.hstack([y_test1, y_test2])

unique_classes     = np.unique(y_test.flatten())
class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y_test.flatten())
class_weights_dict = dict(enumerate(class_weights))
NUM_CLASSES        = len(unique_classes)

preds1 = models[0].predict(X_test, batch_size=256)
# pred2 = np.argmax(preds1, axis=1)

preds2 = models[1].predict(X_test, batch_size=256)
# pred3 = np.argmax(preds3, axis=1)

new_preds = np.concatenate([preds1, preds2], axis=1)
reshaped_new_preds = new_preds.reshape(new_preds.shape[0],1,new_preds.shape[1])

meta_inputs = keras.layers.Input(shape=(1, reshaped_new_preds.shape[-1]))
dim = 64
x = keras.layers.Dense(dim)(meta_inputs)

for _ in range(4):
    mha = keras.layers.MultiHeadAttention(num_heads=4, key_dim=dim*4)(x, x)
    mha = keras.layers.Dropout(0.5)(mha)
    ffn = keras.layers.Dense(dim*4, activation='gelu')(mha)
    ffn = keras.layers.Dense(dim)(ffn)
    ffn = keras.layers.Dropout(0.5)(ffn)
    x   = keras.layers.Add()([ffn, x])
    x   = keras.layers.LayerNormalization()(x)
# x = keras.layers.TimeDistributed(keras.layers.Dense(dim, activation='gelu'))(x)
# x = keras.layers.Dropout(0.5)(x)
# x = keras.layers.GlobalAveragePooling1D()(x)
# x = keras.layers.Dense(dim, activation='gelu')(x)
meta_outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x[:, -1, :])
meta_model = keras.Model(meta_inputs, meta_outputs)
meta_model.summary()

meta_model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, cooldown=0, min_lr=1e-8),
]

meta_model.fit(reshaped_new_preds, y_test, validation_split=0.1, shuffle=True, batch_size=256,
               epochs=EPOCHS, callbacks=callbacks, class_weight=class_weights_dict, verbose=1,
              )
meta_model.save('models/meta_model.keras')

test_data1 = df[df['App'] == 'BETIKA'].copy()
X_test1, y_test1 = create_features(test_data1, WL, TARGET)

test_data2 = df[df['App'] == 'BETGR8'].copy()
X_test2, y_test2 = create_features(test_data2, WL, TARGET)

X_test = np.vstack([X_test1, X_test2])
y_test = np.hstack([y_test1, y_test2])

preds1 = models[0].predict(X_test, batch_size=256)
pred1 = np.argmax(preds1, axis=1)

preds2 = models[1].predict(X_test, batch_size=256)
pred2 = np.argmax(preds2, axis=1)

new_preds1 = np.concatenate([preds1, preds2], axis=1)
reshaped_new_preds1 = new_preds1.reshape(new_preds1.shape[0], 1, new_preds1.shape[1])

preds = meta_model.predict(reshaped_new_preds1, batch_size=256)
pred = np.argmax(preds, axis=1)

print(f'\nAccuracy on Test3 using Meta Model set: {np.mean(pred==y_test):.2f}')

losing_streak1, win_streak1 = calculate_streaks(pred, y_test)

print('\nSample Results')
print(pred[:50])
print(y_test[:50])

print(f'\nLongest losing streak for Test3 using Meta Model  = {losing_streak1}')
print(f'Longest winning streak for Test3 using Meta Model = {win_streak1}')
