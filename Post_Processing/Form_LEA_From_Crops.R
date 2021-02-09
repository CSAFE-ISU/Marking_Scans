df <- df %>% unnest(c(original_indexes, chop_storage))

bm <- df %>% select(Var1, Var2, chop_storage)

bm <- bm[nrow(bm):1, ]



for(i in 1:nrow(bm)){
  bm$surface[[i]] <- bm$chop_storage[[i]]$surface.matrix
  bm$mask[[i]] <- bm$chop_storage[[i]]$mask #as.matrix
  
}

bm <- bm %>% 
  group_by(Var2) %>%
  summarise(matrix_list = list(surface),
            raster_list = list(mask))



rotate_mask_matrix = function(maskM){
  as.raster(apply(t(as.matrix(maskM)), 2, rev))
  
}

rotate_mask_matrix_2 = function(maskM){
  as.raster(apply(as.matrix(maskM), 2, rev))
  
}

for(i in 1:nrow(bm)){
  
  bm$matrix_list_2[[i]] <- do.call(rbind, bm$matrix_list[[i]])
  #bm$raster_list[[i]] <- lapply(bm$raster_list[[i]], rotate_mask_matrix)
  #bm$raster_list[[i]] <- lapply(bm$raster_list[[i]], rotate_mask_matrix)
  bm$raster_list_2[[i]] <- do.call(cbind, bm$raster_list[[i]])
  
}


bm_final <- bm$matrix_list_2
bm_final_2 <- bm$raster_list_2

rotate = function(M){
  t(apply(M, 2, rev))
}


bm_final <- lapply(bm_final, rotate)
bm_final <- lapply(bm_final, rotate)
bm_final_2 <- lapply(bm_final_2, rotate)
bm_final_2 <- lapply(bm_final_2, rotate)

matrix <- do.call(cbind, bm_final) 
raster <- do.call(rbind, bm_final_2)


df$x3p[[1]]$surface.matrix <- matrix
df$x3p[[1]]$mask <- raster


df$x3p[[1]] <- x3p_rotate(df$x3p[[1]], -180)