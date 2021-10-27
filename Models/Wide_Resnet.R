library(keras)
library(tensorflow)

packageVersion("keras")
packageVersion("tensorflow")

initialConvolutionLayer <- function(model, numberOfFilters){
  
  model <- model %>% layer_conv_3d(filters = numberOfFilters,
                                    kernel_size = c(3, 3, 1),
                                    padding = 'same',
                                    kernel_initializer = initializer_he_normal(),
                                    kernel_regularizer = regularizer_l2(weightDecay))
  
  model <- model %>% layer_batch_normalization(axis = -1,
                                               momentum = 0.1,
                                               epsilon = 1.0e-5,
                                               gamma_initializer = "uniform")
  
  model <- model %>% layer_activation_leaky_relu()
  
  return(model)
}

customConvolutionLayer <- function(initialModel,
                                   base,
                                   width,
                                   strides = c(1, 1, 1),
                                   dropoutRate = 0.0,
                                   expand = TRUE){
  
  numberOfFilters <- as.integer(base * width)
  
  if(expand == TRUE){
    model <- initialModel %>% layer_conv_3d(filters = numberOfFilters,
                                            kernel_size = c(3, 3, 1),
                                            padding = 'same',
                                            strides = strides,
                                            kernel_initializer = initializer_he_normal(),
                                            kernel_regularizer = regularizer_l2(weightDecay),
                                            use_bias = FALSE)
  } else {
    model <- initialModel
  }
  
  model <- model %>% layer_batch_normalization(axis = -1,
                                               momentum = 0.1,
                                               epsilon = 1.0e-5,
                                               gamma_initializer = "uniform")
  
  model <- model %>% layer_activation_leaky_relu()
  
  model <- model %>% layer_conv_3d(filters = numberOfFilters,
                                   kernel_size = c( 3, 3, 1 ),
                                   padding = 'same',
                                   kernel_initializer = initializer_he_normal(),
                                   kernel_regularizer = regularizer_l2(weightDecay),
                                   use_bias = FALSE)
  
  if(expand == TRUE){
    skipLayer <- initialModel %>% layer_conv_3d(filters = numberOfFilters,
                                                kernel_size = c( 1, 1, 1 ),
                                                padding = 'same', strides = strides,
                                                kernel_initializer = initializer_he_normal(),
                                                kernel_regularizer = regularizer_l2(weightDecay),
                                                use_bias = FALSE)
    
    model <- layer_add(list(model, skipLayer))
  } else {
    if(dropoutRate > 0.0){
      model <- model %>% layer_dropout(rate = dropoutRate)
    }
    
    model <- model %>% layer_batch_normalization(axis = -1,
                                                 momentum = 0.1,
                                                 epsilon = 1.0e-5,
                                                 gamma_initializer = "uniform")
    
    model <- model %>% layer_activation_leaky_relu()
    
    model <- model %>% layer_conv_3d(filters = numberOfFilters,
                                     kernel_size = c(3, 3, 1),
                                     padding = 'same',
                                     kernel_initializer = initializer_he_normal(),
                                     kernel_regularizer = regularizer_l2(weightDecay),
                                     use_bias = FALSE)
    
    model <- layer_add(list(initialModel, model))
  }
  
  return(model)
}


weightDecay = 0.0005
inputs <- layer_input(shape = c(128, 128, 1, 1))
residualBlockSchedule <- c(8, 16, 32, 64)
width = 10
dropoutRate = 0.0

outputs <- initialConvolutionLayer(inputs, residualBlockSchedule[[1]])

outputs <- customConvolutionLayer(initialModel = outputs,
                                  base = residualBlockSchedule[[1]],
                                  width = width,
                                  strides = c( 1, 1, 1 ),
                                  dropoutRate = 0.0,
                                  expand = TRUE)
for(i in 1:4){
  outputs <- customConvolutionLayer(initialModel= outputs,
                                    base = residualBlockSchedule[[1]],
                                    width = width,
                                    dropoutRate = dropoutRate,
                                    expand = FALSE)
}
outputs <- outputs %>% layer_batch_normalization(axis = -1,
                                                 momentum = 0.1,
                                                 epsilon = 1.0e-5,
                                                 gamma_initializer = "uniform")

outputs <- outputs %>% layer_activation_leaky_relu()


outputs <- customConvolutionLayer(initialModel = outputs,
                                  base = residualBlockSchedule[[2]],
                                  width = width,strides = c(2, 2, 1 ),
                                  dropoutRate = 0.0,
                                  expand = TRUE)
for(i in 1:4){
  outputs <- customConvolutionLayer(initialModel = outputs,
                                    base = residualBlockSchedule[[2]],
                                    width = width,
                                    dropoutRate = dropoutRate,
                                    expand = FALSE)
}
outputs <- outputs %>% layer_batch_normalization(axis = -1,
                                                 momentum = 0.1,
                                                 epsilon = 1.0e-5,
                                                 gamma_initializer = "uniform")

outputs <- outputs %>% layer_activation_leaky_relu()

outputs <- customConvolutionLayer(initialModel = outputs,
                                  base = residualBlockSchedule[[3]],
                                  width = width,
                                  strides = c(2, 2, 1 ),
                                  dropoutRate = 0.0,
                                  expand = TRUE)
for(i in 1:4){
  outputs <- customConvolutionLayer(initialModel = outputs,
                                    base = residualBlockSchedule[[3]],
                                    width = width,
                                    dropoutRate = dropoutRate,
                                    expand = FALSE)
}
outputs <- outputs %>% layer_batch_normalization(axis = -1,
                                                 momentum = 0.1,
                                                 epsilon = 1.0e-5,
                                                 gamma_initializer = "uniform")

outputs <- outputs %>% layer_activation_leaky_relu()


outputs <- customConvolutionLayer(initialModel = outputs,
                                  base = residualBlockSchedule[[4]],
                                  width = width,
                                  strides = c(2, 2, 1 ),
                                  dropoutRate = 0.0,
                                  expand = TRUE)
for(i in 1:4){
  outputs <- customConvolutionLayer(initialModel = outputs,
                                    base = residualBlockSchedule[[4]],
                                    width = width,
                                    dropoutRate = dropoutRate,
                                    expand = FALSE)
}
outputs <- outputs %>% layer_batch_normalization(axis = -1,
                                                 momentum = 0.1,
                                                 epsilon = 1.0e-5,
                                                 gamma_initializer = "uniform")

outputs <- outputs %>% layer_activation_leaky_relu()



outputs <- outputs %>% layer_average_pooling_3d(pool_size = c(8, 8, 1))

outputs <- outputs %>% layer_flatten()

outputs <- outputs %>% layer_dense(units = 6,
                                   kernel_regularizer = regularizer_l2( weightDecay ),
                                   activation = "softmax")

model <- keras_model(inputs = inputs, outputs = outputs)

cat("Wide Resnet Function Compiled\n")