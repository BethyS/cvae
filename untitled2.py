#Losses--------------------------------------------------------------------
    def kl_loss(sample, mean, logstddev, raxis=1):
        """
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian 
        """
        log2pi = tf.math.log(2. * np.pi)
         kl_los= -tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logstddev) + logstddev + log2pi),axis=raxis)
         return kl_los
     
        
    def recon_loss(self,x):
        mean, logstddev = self.encode(x)
        z = self.reparameterize(mean, logstddev)
        x_pred = self.decode(z)
        ms_loss =K.sum(K.square(x - x_pred), axis=-1)
        return ms_loss
    
    
    def clust_loss():
        clus_loss
        return clus_loss
    
    
    def vae_losses(self,ms_loss,kl_los,clus_loss):
        print(recostruction loss:)
        print()
        return ms_loss+kl_los+clus_loss
        
    def compute_loss(model, x):
      mean, logstddev = model.encode(x)
      z = model.reparameterize(mean, logstddev)
      x_logit = model.decode(z)
      cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
      logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
      logpz = log_normal_pdf(z, 0., 0.)
      logqz_x = log_normal_pdf(z, mean, logstddev)
      return -tf.reduce_mean(logpx_z + logpz - logqz_x)
# Optimizer--------------------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(1e-4)   
