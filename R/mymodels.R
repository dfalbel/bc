#' My Random Forest
#'
#' This function implements my random forest tuning paremeters.
#'
#' @param maxit maximum number of iterations in random tune search
#'
#' @export
my_randomForest <- function(maxit = 10){
  base.lrn <- mlr::makeLearner("classif.randomForest", predict.type = "prob")
  ctrl <- mlr::makeTuneControlRandom(maxit = maxit)
  rdesc <- mlr::makeResampleDesc("Holdout")
  par.set <- ParamHelpers::makeParamSet(
    ParamHelpers::makeDiscreteParam("ntree", values = c(100L)),
    ParamHelpers::makeIntegerParam("mtry", lower = 1, upper = 20)
  )
  tuned.lrn <- mlr::makeTuneWrapper(base.lrn, rdesc, mmce, par.set, ctrl)
  tuned.lrn
}
