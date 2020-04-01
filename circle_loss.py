from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class CircleLoss(Model):
	"""This is the tf implementation of the circle loss. It inherits the 
	keras Model classs in tf2. Inputs is a list of three tensors, the first
	one contains the positive feature samples, the second contains the negative
	feature samples, and the third contains the query feartures. """
	
    def __init__(self, scale=32, margin=0.25, similarity='dot', **kwargs):
        self.scale = scale
        self.margin = margin
        self.similarity = similarity
        super(CircleLoss, self).__init__(dynamic=True, **kwargs)
        
    def call(self, inputs):
        p = inputs[0]
        n = inputs[1]
        q = inputs[2]
        
        if self.similarity == 'dot':
            sim_p = self.dot_similarity(q, p)
            sim_n = self.dot_similarity(q, n)
        elif self.similarity == 'cos':
            sim_p = self.cosine_similarity(q, p)
            sim_n = self.cosine_similarity(q, n)       
        else:
            raise ValueError('This similarity is not implemented.')
        
        alpha_p = K.relu(-sim_p + 1 + self.margin)
        alpha_n = K.relu(sim_n + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = K.mean(K.sum(K.exp(-self.scale * alpha_p * (sim_p - margin_p)), axis=1))
        loss_n = K.mean(K.sum(K.exp(self.scale * alpha_n * (sim_n - margin_n)), axis=1))
        return K.log(1 + loss_p * loss_n)
    
    def compute_output_shape(self, input_shape):
        return (1,)
    
    def dot_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        return K.dot(x, K.transpose(y))
    
    def cosine_similarity(self, x, y):
        x = K.reshape(x, (K.shape(x)[0], -1))
        y = K.reshape(y, (K.shape(y)[0], -1))
        abs_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
        abs_y = K.sqrt(K.sum(K.square(y), axis=1, keepdims=True))
        up = K.dot(x, K.transpose(y))
        down = K.dot(abs_x, K.transpose(abs_y))
        return up / down
