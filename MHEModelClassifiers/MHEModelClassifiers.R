############################ INPUT
path = "C:/Users/"
# must contain folder 'datasets' with data sets .txt-format
cl   = 2  # number of clusters for parallel computing

############################ load required packages
pack = c("RRF", "caret", "plyr", "foreach", "gbm", "party", "kknn", "e1071", "kernlab", 
         "RSNNS", "doParallel", "compiler", "randomForest")

lapply(pack, function(x) if (!(x %in% installed.packages())) {
  install.packages(x)
})
lapply(pack, library, character.only = TRUE)

############################ begin

# create directories
dir.create(file.path(path, "models"))
dir.create(file.path(path, "predictions"))
path.data   = paste(path, "/datasets", sep = "")
path.models = paste(path, "/models", sep = "")
path.pred   = paste(path, "/predictions", sep = "")

# global model estimation setup
ctrl = trainControl(method = "cv", number = 5, savePredictions = TRUE, classProbs = TRUE)

# preprocess function
pp = function(data) {
  ind = sapply(data, is.numeric)
  if (sum(ind) == 0) {
    print("There are no numerical features")
  } else {
    data[, ind] = sapply(data[, ind], scale)
  }
  return(data)
}

# load list of data set names
setwd(path.data)
df.list = list.files(path.data)
df.list = gsub("", pattern = ".txt", df.list)

# start modeling process

for (df.ind in 1:length(df.list)) {
  
  print("Prepair Data")
  setwd(path.data)
  df = read.table(paste(path.data, "/", df.list[df.ind], ".txt", sep = ""))
  
  # preprocess data
  class.ind = which(colnames(df)=="class")
  df[,-class.ind] = pp(df[,-class.ind])
  
  # divide into training/testing data
  set.seed(99)
  data.part = createDataPartition(y = df$class, p = 0.7, list = FALSE)
  data.part = as.vector(data.part)
  training  = df[data.part, ]
  testing   = df[-data.part, ]
  
  # save partitioning
  dir.create(file.path(path.pred, df.list[df.ind]))
  setwd(file.path(path.pred, df.list[df.ind]))
  saveRDS(data.part, file = paste(df.list[df.ind], "-", "part.rds", sep = ""))
  
  # save training set observations for later use
  if (is.numeric(df$class) == TRUE) {
    df$class         = factor(df$class) 
    levels(df$class) = c("negative", "positive")
  } 
  obsTrain    = df$class[part]
  obsTrainNum = as.numeric(obsTrain) - 1
  obsTest     = df$class[-part]
  obsTestNum  = as.numeric(obsTest) - 1
  
  write.csv(obsTrain, file = paste(df.list[df.ind], "-obsTrainNum.csv", sep = ""))
  
  # estimation of classifiers
  print("Start Estimation")
  dir.create(file.path(path.models, df.list[df.ind]))
  setwd(file.path(path.models, df.list[df.ind]))
  
  ### svmRadial ###
  print("Modeling svmRadial")
  cost = 10^seq(-6, 3, 1)
  sig  = c(0.01, 0.033, 0.05)
  meth = "svmRadial"
  l    = 1
  
  for (i in 1:length(cost)) {
    for (j in 1:length(sig)) {
      print(l)
      
      tune  = data.frame(.C = cost[i], .sigma = sig[j])
      fl    = makeCluster(cl)
      registerDoParallel(fl)
      model = try(train(form = class ~ ., data = training, method = meth, 
                        tuneGrid = tune, metric = "Accuracy", 
                        maximize = TRUE, trControl = ctrl))
      stopCluster(fl)
      saveRDS(model, file = paste(meth, "-C", cost[i], "-sig", sig[j], ".rds", 
        sep = ""))
      l = l + 1
    }
  }
  
  rm(cost, sig, meth, l, tune, model)
  
  ### svmPoly ###
  print("Modeling svmPoly")
  cost = c(0.25, 0.5, 1)
  sc   = c(0.001, 0.01, 0.1)
  deg  = c(1, 2, 3)
  meth = "svmPoly"
  l    = 1
  
  for (i in 1:length(deg)) {
    for (j in 1:length(sc)) {
      for (k in 1:length(cost)) {
        print(l)
        tune  = data.frame(.degree = deg[i], .scale = sc[j], .C = cost[k])
        fl    = makeCluster(cl)
        registerDoParallel(fl)
        model = try(train(form = class ~ ., data = training, method = meth, 
                          tuneGrid = tune,  metric = "Accuracy", 
                          maximize = TRUE, trControl = ctrl))
        stopCluster(fl)
        saveRDS(model, file = paste(meth, "-deg", deg[i], "-sc", sc[j], "-C", 
          cost[k], ".rds", sep = ""))
        l = l + 1
      }
    }
  }
  rm(cost, sc, deg, meth, l, tune, model)
  
  ### C5.0 Tree ###
  print("Modeling C5.0")
  mod  = c("rules", "tree")
  win  = c(FALSE, TRUE)
  tr   = c(10, 20, 30, 40, 50, 60, 70)
  meth = "C5.0"
  l    = 1
  
  for (i in 1:length(mod)) {
    for (j in 1:length(win)) {
      for (k in 1:length(tr)) {
        print(l)
        tune  = data.frame(.model = mod[i], .winnow = win[j], .trials = tr[k])
        fl    = makeCluster(cl)
        registerDoParallel(fl)
        model = try(train(form = class ~ ., data = training, method = meth, 
                          tuneGrid = tune,  metric = "Accuracy", 
                          maximize = TRUE, trControl = ctrl))
        stopCluster(fl)
        saveRDS(model, file = paste(meth, "-mod", mod[i], "-win", win[j], 
          "-tr", tr[k], ".rds", sep = ""))
        l = l + 1
      }
    }
  }
  rm(mod, win, tr, meth, l, tune, model)
  
  ### Stochastic Gradient Boosting ###
  print("Modeling gbm")
  id   = c(1, 2, 4, 6, 8, 10)
  ntr  = c(100, 200, 300, 400, 500)
  meth = "gbm"
  l    = 1
  
  for (i in 1:length(id)) {
    for (j in 1:length(ntr)) {
      print(l)
      tune  = data.frame(.interaction.depth = id[i], .n.trees = ntr[j], .shrinkage = 0.1, 
        .n.minobsinnode = 10)
      fl    = makeCluster(cl)
      registerDoParallel(fl)
      model = try(train(form = class ~ ., data = training, method = meth, 
                        tuneGrid = tune,  metric = "Accuracy", 
                        maximize = TRUE, trControl = ctrl))
      stopCluster(fl)
      saveRDS(model, file = paste(meth, "-id", id[i], "-ntr", ntr[j], ".rds", sep = ""))
      l = l + 1
    }
  }
  rm(id, ntr, meth, l, tune, model)
  
  ### k-nearest neighbor ###
  print("Modeling knn")
  if (nrow(training) < 30) {
    kseq = seq(3, nrow(training))
  } else kseq = seq(3, 30)
  meth = "knn"
  l    = 1
  
  for (i in 1:length(kseq)) {
    print(l)
    tune  = data.frame(.k = kseq[i])
    fl    = makeCluster(cl)
    registerDoParallel(fl)
    model = try(train(form = class ~ ., data = training, method = meth, 
                      tuneGrid = tune,  metric = "Accuracy", 
                      maximize = TRUE, trControl = ctrl))
    stopCluster(fl)
    saveRDS(model, file = paste(meth, "-k", kseq[i], ".rds", sep = ""))
    l = l + 1
  }
  
  
  rm(kseq, meth, l, tune, model)
  
  ### Multi-Layer Perceptron ###
  print("Modeling mlp")
  s    = seq(1, 19, 2)
  meth = "mlp"
  l    = 1
  
  for (i in 1:length(s)) {
    print(l)
    tune  = data.frame(.size = s[i])
    fl    = makeCluster(cl)
    registerDoParallel(fl)
    model = try(train(form = class ~ ., data = training, method = meth, 
                      tuneGrid = tune,  metric = "Accuracy", 
                      maximize = TRUE, trControl = ctrl))
    stopCluster(fl)
    saveRDS(model, file = paste(meth, "-size", s[i], ".rds", sep = ""))
    l = l + 1
  }
  rm(s, meth, l, tune, model)
  
  ### Neural Network ###
  print("Modeling nnet")
  s    = 2^seq(0, 5)
  dec  = c(0, 0.001, 0.1)
  meth = "nnet"
  l    = 1
  
  for (i in 1:length(s)) {
    for (j in 1:length(dec)) {
      tune  = data.frame(.size = s[i], .decay = dec[j])
      fl    = makeCluster(cl)
      registerDoParallel(fl)
      model = try(train(form = class ~ ., data = training, method = meth, 
                        tuneGrid = tune,  metric = "Accuracy", 
                        maximize = TRUE, trControl = ctrl))
      stopCluster(fl)
      saveRDS(model, file = paste(meth, "-size", s[i], "-dec", dec[j], ".rds", 
        sep = ""))
      l = l + 1
    }
  }
  rm(s, dec, meth, l, tune, model)
  
  ### Regularized Random Forest ###
  print("Modeling RFF")
  mt   = c(0.5, 1, 2) * floor(sqrt(ncol(training)))
  reg  = c(0.01, 0.505, 1)
  imp  = c(0, 0.5, 1)
  meth = "RRF"
  l    = 1
  
  for (i in 1:length(mt)) {
    for (j in 1:length(reg)) {
      for (k in 1:length(imp)) {
        print(l)
        tune  = data.frame(.mtry = mt[i], .coefReg = reg[j], .coefImp = imp[k])
        fl    = makeCluster(cl)
        registerDoParallel(fl)
        model = try(train(form = class ~ ., data = training, method = meth, 
                          tuneGrid = tune,  metric = "Accuracy", 
                          maximize = TRUE, trControl = ctrl))
        stopCluster(fl)
        saveRDS(model, file = paste(meth, "-mtry", mt[i], "-creg", reg[j], 
          "-cimp", imp[k], ".rds", sep = ""))
        l = l + 1
      }
    }
  }
  rm(mt, reg, imp, meth, l, tune, model)
  
  ############################ make and stack predictions
  
  print(paste("Make predictions for ", df.list[df.ind], sep = ""))
  
  mod       = list.files()
  mod.names = gsub(".rds", "", mod)
  k         = 1
  
  for (j in 1:length(mod)) {
    print(j)
    model = readRDS(mod[j])
    pred  = try(model$pred[order(model$pred$rowIndex), ], silent = TRUE)
    
    if (class(pred) == "try-error") {
      unlink(mod[j])
      next
    } else if (k == 1) {
      train.frame = data.frame(pred$positive)
      test.frame  = data.frame(predict(model, newdata = testing, type = "prob")$positive)
      
      colnames(train.frame) = mod.names[j]
      colnames(test.frame)  = mod.names[j]
      
      k = k + 1
    } else {
      train.frame = cbind(train.frame, pred$positive)
      test.frame  = cbind(test.frame, predict(model, newdata = testing, type = "prob")$positive)
      
      colnames(train.frame)[k] = mod.names[j]
      colnames(test.frame)[k]  = mod.names[j]
      
      k = k + 1
    }
  }
  
  setwd(file.path(path.pred, df.list[df.ind]))
  saveRDS(train.frame, file  = paste(df.list[df.ind], "-predTrain.rds", sep = ""))
  saveRDS(test.frame, file   = paste(df.list[df.ind], "-predTest.rds", sep = ""))
  write.csv(train.frame,file = paste(df.list[df.ind], "-predTrain.csv", sep = ""))
  write.csv(test.frame,file  = paste(df.list[df.ind], "-predTest.csv", sep = ""))
  
  rm(test.frame, train.frame, model, pred, k)
}
