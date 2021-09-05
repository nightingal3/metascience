library(lme4)
library(caret)
library(sjPlot)
library(brms)
library(magrittr)
library(tidybayes)
library(modelr)
library(dplyr)

run_model <- function(expr, path, reuse = TRUE) {
    path <- paste0(path, ".Rds")
    if (reuse) {
        fit <- suppressWarnings(try(readRDS(path), silent = TRUE))
    }
    if (is(fit, "try-error")) {
        fit <- eval(expr)
        saveRDS(fit, file = path)
    }
    fit
}

df <- read.csv("data/merged_data.csv")

model <- run_model(brms::brm("ll_diff_proto ~ 1 + num_papers*median_collaborators + (1 + num_papers*median_collaborators|field)", data=df, save_all_pars = TRUE), path="./fit-model-proto")
model_no_interaction <- run_model(brms::brm("ll_diff_proto ~ 1 + num_papers + median_collaborators + (1 + num_papers + median_collaborators|field)", data=df, save_all_pars = TRUE), path="./fit-model-proto-1")
model_only_num_papers <- run_model(brms::brm("ll_diff_proto ~ 1 + num_papers + (1 + num_papers |field)", data=df, save_all_pars = TRUE), path="./fit-model-proto-2")
model_only_num_collabs <- run_model(brms::brm("ll_diff_proto ~ 1 + median_collaborators + (1 + median_collaborators |field)", data=df, save_all_pars = TRUE), path="./fit-model-proto-3")
model_only_year <- run_model(brms::brm("ll_diff_1NN ~ 1 + oldest_paper + (1 + oldest_paper |field)", data=df, save_all_pars = TRUE), path="./fit-model-proto-4")
#model_only_num_papers <- run_model(brms:brm("ll_diff_1NN ~ 1 + num_papers + (1 + num_papers|field)", data=df, save_all_pars = TRUE), path="./fit-model-2")
#model_only_num_collabs <- run_model(brms:brm("ll_diff_1NN ~ 1 + num_collaborators + (1 + num_collaborators|field)", data=df, save_all_pars = TRUE), path="./fit-model-3")

#preds <- predict(model)
#print(preds)
#print(summary(model))
#print(summary(model_no_interaction))
#print(summary(model_only_num_papers))
print(summary(model_only_num_collabs))
#print(LOO(model, model_no_interaction, model_only_num_papers, model_only_num_collabs, model_only_year))
#print(LOO(model_no_gender, model_no_interaction))


#model %>%
#spread_draws(b_bert_pred, b_gender, r_assessor_id[condition,Intercept], n=5) %>%
#mutate(condition_mean = b_bert_pred+b_gender+r_assessor_id) %>%
#print() 
#ggplot(aes(y = condition, x = condition_mean)) +
#stat_halfeye() +
#ggsave("plot.png")
