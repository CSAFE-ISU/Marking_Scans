
set.seed(777)

library(keras)
library(tensorflow)
library(tidyverse)
library(abind)



cnn_model_2 <- keras_model_sequential() %>%
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
  callback_model_checkpoint(filepath = "my_model.h5", monitor = "val_loss",save_best_only = TRUE),
  callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 10))



cnn_model_2 %>% compile(optimizer = "rmsprop",
                      loss = "categorical_crossentropy",
                      metrics = c("accuracy"))


x_test_2 <- array_1_1[1:2000, , ,1, 1]
x_test <- array(c(x_test),
                dim = c(2000, 128, 128, 1, 1))

y_test <- y_1_1[1:2000,]


cnn_model_2 %>% fit(array_1_0, y_1_0, 
                    batch_size = 100,
                    epochs = 5,
                    validation_data = list(x_test, y_test),
                    callbacks = callbacks_list)



df <- as.data.frame(rbind(y_1_0, y_1_1, y_1_2, y_1_3, y_1_4))

head(df)

#V1 <- Good striation
#V2 <- Right & Left Groove
#V3 <- Vertical Striations
#V4 <- No Striation
#V5 <- Damage
#V6 <- Breakoff

df <- df %>%
  rename(Well_Expressed_Striae = V1,
         Grooves = V2,
         Vertical_Striae = V3,
         No_Striae = V4,
         Damage = V5,
         Breakoff = V6)


df <- gather(df, Annotation, Value, Well_Expressed_Striae:Breakoff)

df %>% group_by(Annotation) %>%
  summarise(sum = sum(Value)) %>%
    ggplot(aes(x = Annotation, y = sum))+
      geom_bar(stat = "identity", fill = "orange")+
        theme_bw()+ xlab("Marking Scan Annotations")+ylab("# of Occurences")+ggtitle("Class Imbalance")
