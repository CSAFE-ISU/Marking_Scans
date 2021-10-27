library(keras)
library(tensorflow)

packageVersion("keras")
packageVersion("tensorflow")

#Residual Network 18 Architecture

create_residual_3dCNN_V2 <- function(Learning_rate = 0.1, decay = 0.0001){
  
  k_clear_session()
  tensorflow::tf$random$set_seed(777777)
  
  input <- layer_input(shape = c(128, 128, 1, 1))
  
  skip_1 <- input %>% #-----------------------------------------------------------------------------> First skip connection being usec
    layer_conv_3d(filters = 32, kernel_size = c(7, 7, 1), strides = c(2, 2, 1), padding = "same") %>% #Layer 2
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_max_pooling_3d(pool_size = c(2, 2, 1))
  
  output_1 <- skip_1 %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% #Layer 2
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% #Layer 3
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_2 <- layer_add(list(output_1, skip_1)) %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same")%>% #Layer 4
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 5
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_3 <- layer_add(list(output_2, output_1)) %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same")%>% # Layer 6
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 32, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 7
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_4 <- layer_add(list(output_3, output_2)) %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(2, 2, 1), padding = "same") %>% # Layer 8
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # layer 9
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  skip_2 <- output_3 %>% #-----------------------------------------------------------------------------> feature_maps differ,so we perform a projection using 1x1 convolution
    layer_conv_3d(filters = 64, kernel_size = 1, strides = c(2,2,1), padding = "same") %>% # Layer 10
    layer_batch_normalization(center = TRUE, scale = TRUE) 
  
  
  output_5 <- layer_add(list(output_4, skip_2)) %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 11
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 12
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_6 <- layer_add(list(output_5, output_4)) %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 13
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 14
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_7 <- layer_add(list(output_6, output_5)) %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 15
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 64, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 16
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_8 <- layer_add(list(output_7, output_6)) %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(2, 2, 1), padding = "same") %>% # Layer 17
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 18
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  skip_3 <- output_7 %>% #-----------------------------------------------------------------------------> feature_maps differ,so we perform a projection using 1x1 convolution
    layer_conv_3d(filters = 128, kernel_size = 1, strides = c(2, 2, 1), padding = "same") %>% # Layer 19
    layer_batch_normalization(center = TRUE, scale = TRUE)
  
  output_9 <- layer_add(list(output_8, skip_3)) %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 20
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 21
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_10 <- layer_add(list(output_9, output_8)) %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 22
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 128, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 23
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output_11 <- layer_add(list(output_10, output_9)) %>%
    layer_conv_3d(filters = 256, kernel_size = c(3, 3, 1), strides = c(2, 2, 1), padding = "same") %>% # Layer 24
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 256, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 25
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  skip_4 <- output_10 %>%
    layer_conv_3d(filters = 256, kernel_size = 1, strides = c(2, 2, 1), padding = "same") %>% # Layer 26
    layer_batch_normalization(center = TRUE, scale = TRUE)
  
  output_12 <- layer_add(list(output_11, skip_4)) %>%
    layer_conv_3d(filters = 256, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 27
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_conv_3d(filters = 256, kernel_size = c(3, 3, 1), strides = c(1, 1, 1), padding = "same") %>% # Layer 28
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu")
  
  output <- output_12 %>%
    layer_global_average_pooling_3d() %>%
    layer_flatten() %>%
    layer_dense(units = 512) %>% #Layer 29
    layer_batch_normalization(center = TRUE, scale = TRUE) %>%
    layer_activation("relu") %>%
    layer_dense(units = 6, activation = "softmax") # Layer 30
  
  
  
  cnn_resNet_30 <- keras_model(input, output)
  
  cnn_resNet_30 %>% compile(loss = "categorical_crossentropy",
                            optimizer = optimizer_rmsprop(lr = Learning_rate, decay = decay),
                            metrics = "accuracy")
  
  return(cnn_resNet_30)
  
  
}


model <- create_residual_3dCNN_V2()
cat("Resnet Function Compiled\n")