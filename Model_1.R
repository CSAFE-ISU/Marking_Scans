
set.seed(777777) # Set Large Seed.  Seed must be set before loading keras & Tensorflow

library(keras)
library(tensorflow)
library(tidyverse)
library(abind)



cnn_model <- keras_model_sequential() %>%
  layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), input_shape = c(128, 128, 1, 1)) %>%
  layer_zero_padding_3d() %>%
  layer_activation_relu() %>%
  layer_max_pooling_3d(pool_size = c(2, 2, 2)) %>%
  layer_batch_normalization(center = TRUE, scale = TRUE) %>%
  layer_spatial_dropout_3d(0.5) %>%
  layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1)) %>%
  layer_zero_padding_3d() %>%
  layer_activation_relu() %>%
  layer_max_pooling_3d(pool_size = c(2, 2, 2)) %>%
  layer_batch_normalization(center = TRUE, scale = TRUE) %>%
  layer_spatial_dropout_3d(0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 256) %>%
  layer_activation_relu() %>%
  layer_dense(units = 256) %>%
  layer_activation_relu() %>%
  layer_dense(units = 6, activation = "softmax")



callbacks_list <- list(
  callback_model_checkpoint(filepath = "my_model.h5", monitor = "val_loss",save_best_only = TRUE),# Save the best Model out of all epochs
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 10)) # Adjust Learning rate if there is no improvement



cnn_model %>% compile(optimizer = "rmsprop",
                      loss = "categorical_crossentropy",
                      metrics = c("accuracy"))



array_1_x <- abind(array_1_0, array_1_1, array_1_2, array_1_3, along=1) # Full Training Data.  

y_1_x <- rbind(y_1_0, y_1_1, y_1_2, y_1_3) # Full Training Data




cnn_model %>% fit(array_1_x, y_1_x, 
                    batch_size = 100,
                    epochs = 5,
                    validation_data = list(array_1_4, y_1_4), # Data used for Validation
                    callbacks = callbacks_list)


