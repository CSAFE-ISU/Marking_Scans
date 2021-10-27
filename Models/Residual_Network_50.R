library(keras)
library(tensorflow)

packageVersion("keras")
packageVersion("tensorflow")

#Dynamic Residual Network
#This function allows a user to create a 3d convolutional residual network
#The network can be as deep as the user wants
#The user is also able to create the Residual_X architecture by changing the inputScalarSize and the cardinality

#inputImageSize, the size of your input tensor
#inputScalerSize, scale parameter of your input shape, we do not use this for our project
#numberOfClassificationLabels, the number of labels we are trying to predict
#layers, the amount of residual blocks
#residualBlockSchedule, the number of sub residual blocks
#lowestResolution, the smallest filter size
#mode, classification or regression, regression has not been implemented, classification assumes non-binary problem

createResNetModel3D <- function(inputImageSize,
                                 inputScalarsSize = 0,
                                 numberOfClassificationLabels = 6,
                                 layers = 1:4,
                                 residualBlockSchedule = c(3, 4, 6, 3),
                                 lowestResolution = 64,
                                 cardinality = 1,
                                 mode = c('classification')){
  
  addCommonLayers <- function(model){
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()
    return(model)
  }
  
  groupedConvolutionLayer3D <- function(model, numberOfFilters, strides){
    
    # Per standard ResNet, this is just a 3-D convolution
    if(cardinality == 1){
      groupedModel <- model %>% layer_conv_3d(filters = numberOfFilters,
                                              kernel_size = c(3, 3, 1),
                                              strides = strides,
                                              padding = 'same')
      return(groupedModel)
    }
    
    if(numberOfFilters %% cardinality != 0)
    {
      stop("Possible Filter Issue")
    }
    
    numberOfGroupFilters <- as.integer(numberOfFilters / cardinality)
    convolutionLayers <- list()
    
    for(j in 1:cardinality){
      convolutionLayers[[j]] <- model %>% layer_lambda(function( z )
      {
        k_set_image_data_format('channels_last')
        z[,,,,((j - 1) * numberOfGroupFilters + 1 ):(j * numberOfGroupFilters)]
      })
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>%
        layer_conv_3d(filters = numberOfGroupFilters,
                      kernel_size = c(3, 3, 1),
                      strides = strides,
                      padding = 'same')
    }
    groupedModel <- layer_concatenate(convolutionLayers)
  
    return(groupedModel)
  }
  
  residualBlock3D <- function(model, 
                              numberOfFiltersIn,
                              numberOfFiltersOut,
                              strides = c(1, 1, 1),
                              project = FALSE){
    shortcut <- model
    
    model <- model %>% layer_conv_3d(filters = numberOfFiltersIn,
                                     kernel_size = c(1, 1, 1),
                                     strides = c(1, 1, 1),
                                     padding = 'same')
    
    model <- addCommonLayers(model)
    
    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer3D(model,
                                       numberOfFilters = numberOfFiltersIn,
                                       strides = strides)
    model <- addCommonLayers(model)
    
    model <- model %>% layer_conv_3d(filters = numberOfFiltersOut,
                                     kernel_size = c(1, 1, 1),
                                     strides = c(1, 1, 1),
                                     padding = 'same')
    
    model <- model %>% layer_batch_normalization()
    
    if(project == TRUE || prod(strides == c(1, 1, 1)) == 0)
    {
      shortcut <- shortcut %>% layer_conv_3d(filters = numberOfFiltersOut,
                                             kernel_size = c(1, 1, 1),
                                             strides = strides,
                                             padding = 'same')
      
      shortcut <- shortcut %>% layer_batch_normalization()
    }
    
    model <- layer_add(list( shortcut, model ))
    
    model <- model %>% layer_activation_leaky_relu()
    
    return(model)
  }
  
  mode <- match.arg(mode)
  
  inputImage <- layer_input(shape = inputImageSize)
  
  nFilters <- lowestResolution
  
  outputs <- inputImage %>% layer_conv_3d(filters = nFilters,
                                          kernel_size = c(7, 7, 1),
                                          strides = c(2, 2, 1),
                                          padding = 'same')
  
  outputs <- addCommonLayers(outputs)
  
  outputs <- outputs %>% layer_max_pooling_3d(pool_size = c(3, 3, 1),
                                              strides = c(2, 2, 1),
                                              padding = 'same')
  
  for(i in seq_len(length(layers))){
    
    nFiltersIn <- lowestResolution * 2 ^ (layers[i])
    
    nFiltersOut <- 2 * nFiltersIn
    
    for(j in seq_len(residualBlockSchedule[i])){
      
      project <- FALSE
      
      if(i == 1 && j == 1){
        
        project <- TRUE
        
      }
      if(i > 1 && j == 1){
        strides <- c(2, 2, 1)
      } else {
        strides <- c(1, 1, 1)
      }
      outputs <- residualBlock3D(outputs,
                                 numberOfFiltersIn = nFiltersIn,
                                 numberOfFiltersOut = nFiltersOut,
                                 strides = strides,
                                 project = project)
    }
  }
  outputs <- outputs %>% layer_global_average_pooling_3d()
  
  layerActivation <- ''
  if(mode == 'classification'){
    
    layerActivation <- 'softmax'
    
  } else {
    
    stop('Regression is not implemented in this function, please use Classification or leave... lol')
  }
  
  resNetModel <- NULL
  
  if(inputScalarsSize > 0){
    
    inputScalars <- layer_input(shape = c(inputScalarsSize))
    
    concatenatedLayer <- layer_concatenate(list(outputs, inputScalars))
    
    outputs <- concatenatedLayer %>%
      layer_dense(units = numberOfClassificationLabels, activation = layerActivation)
    
    resNetModel <- keras_model(inputs = list(inputImage, inputScalars), outputs = outputs)
    
  } else {
    outputs <- outputs %>%
      layer_flatten() %>%
      layer_dense(units = numberOfClassificationLabels, activation = layerActivation)
    resNetModel <- keras_model(inputs = inputImage, outputs = outputs)
  }
  
  return(resNetModel)
}

  
model <- createResNetModel3D(inputImageSize = c(128, 128, 1, 1),
                               inputScalarsSize = 0,
                               numberOfClassificationLabels = 6,
                               layers = 1:4,
                               residualBlockSchedule = c(3, 4, 6, 3),
                               lowestResolution = 64,
                               cardinality = 1,
                               mode = 'classification')  
  

model
cat("Resnet Function Compiled\n")