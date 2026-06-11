import tensorflow as tf

def preprocess_image(image, label, img_size=(32, 32)):
    """
    Resizes and normalizes the input image for the CNN model.
    """
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0  # Scale pixels to [0, 1]
    return image, label

def prepare_dataset(dataset, batch_size=64, is_training=True):
    """
    Optimizes the tf.data pipeline with caching and prefetching.
    """
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
