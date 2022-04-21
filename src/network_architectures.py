
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, \
    Conv2D, GaussianNoise, Input, MaxPool2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from network_components import dilated_block, dilated_residual_block, res_bottlneck,res_block
from tensorflow.keras import backend as K

def residual_u_net_2d(
        activation='selu',
        depth=5,
        filters=16,
        final_activation='selu',
        input_dims=(None, None, 1),
        loss=None,
        merge=Add(),
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
):
    """A U-Net with residual blocks at each level.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        merge:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Residual U-Net.
    """
    if loss is None:
        loss = 'IOU'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)
    pred = res_block(
            pred,
            filter_shape=(7, 7),
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)
        
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)

        crosses.append(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = res_bottlneck(
        pred,
        filters=filters,
        activation=activation,
        project=True,
        merge=merge)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = Concatenate()([pred, cross])
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)
#        pred = res_bottlneck(
#            pred,
#            filters=filters,
#            activation=activation,
#            project=True,
#            merge=merge)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model
