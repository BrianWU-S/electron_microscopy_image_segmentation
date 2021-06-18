from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

act = "relu"
dropout_rate = 0.5
padding = "same"
kernel_initialization = "he_normal"


def UNet(pretrained_weights=None, input_size=(256, 256, 1), lr=0.0005):
    """
    definition of Unet architecture
    :param pretrained_weights: whether there is a pretrained model
    :param input_size:the size and channel of input figure
    :param lr: initial learning rate
    """
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(inputs)
    conv1 = Conv2D(64, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(pool1)
    conv2 = Conv2D(128, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(pool2)
    conv3 = Conv2D(256, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(pool3)
    conv4 = Conv2D(512, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(pool4)
    conv5 = Conv2D(1024, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv5)
    drop5 = Dropout(dropout_rate)(conv5)
    
    up6 = Conv2D(512, 2, activation=act, padding=padding, kernel_initializer=kernel_initialization)(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(merge6)
    conv6 = Conv2D(512, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv6)
    
    up7 = Conv2D(256, 2, activation=act, padding=padding, kernel_initializer=kernel_initialization)(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(merge7)
    conv7 = Conv2D(256, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv7)
    
    up8 = Conv2D(128, 2, activation=act, padding=padding, kernel_initializer=kernel_initialization)(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(merge8)
    conv8 = Conv2D(128, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv8)
    
    up9 = Conv2D(64, 2, activation=act, padding=padding, kernel_initializer=kernel_initialization)(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(merge9)
    conv9 = Conv2D(64, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv9)
    conv9 = Conv2D(2, 3, activation=act, padding=padding, kernel_initializer=kernel_initialization)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(inputs=inputs, outputs=conv10)
    
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    
    return model


if __name__ == '__main__':
    model = UNet()
    model.summary()
