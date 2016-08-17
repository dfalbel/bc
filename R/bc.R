#' Create Task (Define Data)
#'
#' @param data dataset used to train
#' @param target name of response variable
#' @param perc the percentage of a feature values in [0,1) that must differ from the mode value
#' @param beta response rate in original dataset (number of responses/number of non-responses)
#'
#' @export
create_task <- function(data, target, perc = 0.01, beta){
  task <- mlr::makeClassifTask(
    id = "task",
    data = data,
    target = target,
    check.data = TRUE
    )
  task <- mlr::removeConstantFeatures(task, perc = perc, show.info = T)
  task$beta <- beta
  return(task)
}

#' Train Model
#'
#' This functions trains a model defined by a
#' mlr classifier. This is really just a wrapper mlr.
#' It carries the beta parameter that will be used in the
#' probability estimation later.
#'
#' Its important to say that performance
#'
#' @param task the task object
#' @param model the model object
#' @param perc_perf percentage of task size to be used to calculate performance
#'
#' @export
train_model <- function(task, model, perc_perf = 0.2){
  n <- mlr::getTaskSize(task)
  subset <- sample(1:n, size = floor((1 - perc_perf)*n))
  trained_model <- mlr::train(model, task, subset = subset)
  # perf <- predict(trained_model, task, subset = (1:n)[!(1:n) %in% subset]) %>%
  #   performance()
  perf <- model_performance(task$env$data[(1:n)[!(1:n) %in% subset], ], model = trained_model, transform_prob = F)
  trained_model$beta <- task$beta
  trained_model$performance <- perf
  return(trained_model)
}

#' Model Performance
#'
#' Calculates the model performance in a data-set
#'
#' @param data data-set to calculate performance
#' @param model model to calculate performance
#' @param transform_prob should probability be transformed before calculating performance
#' metrics?
#'
#' @export
model_performance <- function(data, model, transform_prob){
  predict_model(data, model,  transform_prob = transform_prob) %>%
    performance(measures = list(mlr::mmce, mlr::auc, mlr::bac))
}

#' Predict Model
#'
#' @param data dataset to predict
#' @param model model object
#' @param transform_prob should probability be transformed using the beta parameter?
#'
#' @export
predict_model <- function(data, model, transform_prob = T){
  pred <- predict(model, newdata = data)
  if (transform_prob) {
    pred$data$prob.1 <- model$beta*pred$data$prob.1/(model$beta*pred$data$prob.1 - pred$data$prob.1 + 1)
    pred$data$prob.0 <- 1 - pred$data$prob.1
    pred$threshold <- model$beta*pred$threshold/(model$beta*pred$threshold - pred$threshold + 1)
  }
  return(pred)
}


