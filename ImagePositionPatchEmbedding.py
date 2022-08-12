import os
# Suppress the INFO message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers


class ImagePosEmbed(layers.Layer):
    def __init__(self, batch_size, patch_num, proj_dim):
        """
        Initialize ImagePosEmbed class which constructs the combined image plus position embeddings layer

        Parameters
        ----------
        patch_size : integer
          The width and height of the patch
        patch_num : integer
          Number of total patches
        proj_dim : integer
          The vector size to project each patch to
        """
        super(ImagePosEmbed, self).__init__()
        self.batch_size = batch_size
        self.patch_num = patch_num
        self.proj_dim = proj_dim
        self.img_embedding_layer = layers.Dense(units=proj_dim)
        self.class_token = self.learnable_class_token()
        self.pos_embedding = self.learnable_pos_embedding()

    def call(self, images):
        """
        When called split the image into patches and generate embeddings for each patch. Concat the class token
        with the patches and add it with the position embeddings to get the final image plus position embeddings
        of each patch and class token.

        Parameters
        ----------
        images : tensor
          The tensor of the images

        Returns
        -------
        embeddings : tensor
          The image plus position embeddings of each patch and class token
        """
        multi_class_token = tf.broadcast_to(self.class_token, [self.batch_size, 1, self.proj_dim])
        embeddings = tf.concat([multi_class_token, self.img_embedding_layer(images)], 1) + self.pos_embedding
        return embeddings


    def learnable_pos_embedding(self):
        """
        Creates learnable position embeddings to all positions

        Returns
        -------
        position_embedding : tensor
          Position embeddings of all positions
        """
        positions = tf.range(start=0, limit=self.patch_num + 1, delta=1)
        position_embedding_layer = layers.Embedding(
            input_dim=self.patch_num + 1, output_dim=self.proj_dim
        )
        position_embedding = position_embedding_layer(positions)

        return position_embedding

    def learnable_class_token(self):
        """
        Creates a learnable class token to represent the whole image

        Returns
        -------
        class_token : tensor
          The class token
        """
        class_token_layer = layers.Embedding(
            input_dim=1, output_dim=self.proj_dim,
        )
        class_token = class_token_layer(0)

        return class_token
