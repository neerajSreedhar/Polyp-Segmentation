import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, Activation, UpSampling2D, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model

def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def build_model():
    size = 256
    num_filters = [64, 128, 256, 512]
    inputs = Input(shape=(size, size, 3))

    skip_x = []
    x = inputs

    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPooling2D((2,2))(x)

    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()

    for i, f, in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    x = Conv2D(1, (1, 1), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(inputs, x)

if __name__ == '__main__':
    model = build_model()
    model.summary()