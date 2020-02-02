import tensorflow as tf
LR_SIZE = 34
HR_SIZE = 96


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def l1_loss(y_true, y_pred, y_mask):
    """
    Modified l1 loss to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = tf.cast(
                cropped_predictions, tf.float32)*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            l1_loss = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.abs(
                    tf.subtract(cropped_labels_masked,
                                corrected_cropped_predictions)
                ), axis=[1, 2]
            )
            X.append(l1_loss)
    X = tf.stack(X)
    min_l1 = tf.reduce_min(X, axis=0)

    return min_l1

def psnr(y_true, y_pred, y_mask):
    """
    Modified PSNR metric to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = tf.cast(
                cropped_predictions, tf.float32)*cropped_y_mask
            cropped_labels_masked = tf.cast(
                cropped_labels, tf.float32)*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            corrected_mse = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.square(
                    tf.subtract(cropped_labels_masked,
                                corrected_cropped_predictions)
                ), axis=[1, 2])

            cPSNR = 10.0*log10((65535.0**2)/corrected_mse)
            X.append(cPSNR)

    X = tf.stack(X)
    max_cPSNR = tf.reduce_max(X, axis=0)  
    return tf.reduce_mean(max_cPSNR)


def loss_mmse(y_true, y_pred, y_mask):
    """
    Modified MSE loss to take into account pixel shifts
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = cropped_predictions*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask

            corrected_mse = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.square(
                    tf.subtract(cropped_labels_masked,
                                corrected_cropped_predictions)
                ), axis=[1, 2])
            X.append(corrected_mse)

    X = tf.stack(X)
    minim = tf.reduce_min(X, axis=0)
    return minim


def charbonnier_loss(y_true, y_pred, y_mask):
    """
    Compute the generalized charbonnier loss of the difference tensor x.
    """
    y_shape = tf.shape(y_true)
    border = 3
    max_pixels_shifts = 2*border
    # crop the center of the reconstructed HR image
    size_image = HR_SIZE
    size_croped_image = size_image - max_pixels_shifts
    clear_pixels = size_croped_image*size_croped_image
    cropped_predictions = y_pred[:, border:size_image -
                                 border, border:size_image-border]
    alpha = 0.45
    beta = 1.0
    epsilon = 0.001

    X = []
    for i in range(max_pixels_shifts+1):  # range(7)
        for j in range(max_pixels_shifts+1):  # range(7)
            cropped_labels = y_true[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]
            cropped_y_mask = y_mask[:, i:i+(size_image-max_pixels_shifts),
                                    j:j+(size_image-max_pixels_shifts)]

            cropped_y_mask = tf.cast(cropped_y_mask, tf.float32)

            cropped_predictions_masked = cropped_predictions*cropped_y_mask
            cropped_labels_masked = cropped_labels*cropped_y_mask

            total_pixels_masked = tf.reduce_sum(cropped_y_mask, axis=[1, 2])

            # bias brightness
            b = (1.0/total_pixels_masked)*tf.reduce_sum(
                tf.subtract(cropped_labels_masked, cropped_predictions_masked),
                axis=[1, 2])

            b = tf.reshape(b, [y_shape[0], 1, 1, 1])

            corrected_cropped_predictions = cropped_predictions_masked+b
            corrected_cropped_predictions = corrected_cropped_predictions*cropped_y_mask
            diff = tf.subtract(cropped_labels_masked,
                               corrected_cropped_predictions)

            error = tf.pow(tf.square(diff * beta) + tf.square(epsilon), alpha)
            error = tf.multiply(cropped_y_mask, error)

            X.append(tf.reduce_sum(error) / total_pixels_masked)

    X = tf.stack(X)
    # Take the minimum mse
    minim = tf.reduce_min(X, axis=0)
    return minim
