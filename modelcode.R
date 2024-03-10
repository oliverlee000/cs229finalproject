library(readr)
library(aod)
library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(caret)
library(reshape2)
library(foreign)
library(nnet)
library(glmnet)
library(neuralnet)
library(Rtsne)
library(caret)
library(VGAM)
library(Metrics)
library(DPpack)
library(ggbeeswarm)
library(rpart)
ploterrors = function(){
predicted_probs <- predict(actual, newdata = df[1:cutoff2,], type = "probs")
semrel_dummy = model.matrix(~ semrel_f - 1, data = df[1:cutoff2,])
cross_entropy_per_observation <- rowSums(loss.cross.entropy(predicted_probs, semrel_dummy))
traindev <- df[1:cutoff2,] %>% mutate(loss = cross_entropy_per_observation) %>% select(stim, category1f, category2f, cat12, loss) 
ggplot(data = traindev[-433,], mapping = aes(x = cat12, y = loss)) +
  geom_boxplot() +
  labs(x = "categories")}
evaluate = function(a, y) {
  for (x in a){
  # Compute accuracy for train set
  predictions <- predict(x, newdata = train, type = "class")
  predictions <- factor(predictions, levels = levels(train[[y]]))
  confusion_matrix <- confusionMatrix(predictions, train[[y]])
  accuracy <- confusion_matrix$overall["Accuracy"]
  print(paste("Accuracy for train set:", accuracy))
  # Compute accuracy for dev set
  predictions <- predict(x, newdata = test, type = "class")
  predictions <- factor(predictions, levels = levels(test[[y]]))
  confusion_matrix <- confusionMatrix(predictions, test[[y]])
  accuracy <- confusion_matrix$overall["Accuracy"]
  print(paste("Accuracy for test set:", accuracy))
  }
  }
setwd('/Users/oliverlee/Documents/cs229project')
df_raw <- read_csv("./LADECv1-2019.csv")
vecs_raw <- read_delim("./glove.6B/glove.6B.50d.txt", quote = "", col_names = FALSE, n_max = Inf, delim = " ", skip = 1)
df <- df_raw %>% drop_na(relation) %>% drop_na(category1) %>% filter(substr(c2, str_length(relation) - 1, str_length(relation) - 1) != "s")
vecs <- vecs_raw %>% filter(X1 %in% df$c1)
vecs <- rbind(vecs, vecs_raw %>% filter(X1 %in% df$c2)) %>% distinct()
vecs = cbind(vecs[,1], prcomp(vecs[,2:51], rank. = 15, center = TRUE, scale. = TRUE)$x)
df <- df %>% inner_join(vecs, by = join_by(c1 == X1), suffix = c("", ".c1"))
df <- df %>% left_join(vecs, by = join_by(c2 == X1), suffix = c(".c1", ".c2"), keep = TRUE)
df = df[sample(1:nrow(df)), ] 
df = df %>% mutate(c1factor = as.factor(c1))
df = df %>% mutate(c2factor = as.factor(c2))
df = df %>% mutate(category1f = as.factor(category1))
df = df %>% mutate(category2f = as.factor(category2))
df = df %>% mutate(cat12 = paste(category1, category2))
df = df %>% mutate(semrel_f = as.factor(case_when(
  relation == 1 ~0,
  relation %in% c(2, 3, 12, 15, 16) ~1,
  relation == 4 ~2,
  relation %in% c(5, 13) ~3,
  relation == 6 ~4,
  relation %in% c(7, 21, 22) ~5,
  relation == 8 ~6,
  relation %in% c(9, 10) ~7,
  relation == 11 ~8,
  relation == 14 ~9,
  relation == 17 ~10,
  relation == 18 ~11,
  relation == 19 ~12,
  relation == 20 ~13)))
df = df %>% mutate(metrel_f = as.factor(case_when(
  relation == 1 ~0,
  relation %in% c(2, 3, 4, 12, 15, 16, 20) ~1,
  relation %in% c(11, 19) ~2,
  relation %in% c(5, 6, 7, 9, 10, 13, 13, 14, 21, 22) ~3,
  relation %in% c(8, 14, 18) ~4)))
df = df %>% mutate(model.matrix( ~ c1 - 1, data = df))
df = df %>% mutate(model.matrix( ~ c2 - 1, data = df))
cutoff1 = floor(0.8 * nrow(df))
cutoff2 = floor(0.9 * nrow(df))
train = df[1:cutoff1, ]
dev = df[cutoff1:cutoff2, ]
test = df[cutoff2:nrow(df), ]
# Create baseline models
if(FALSE){
print("trying cross validation")
num_folds = 4
cv <- trainControl(method = "cv", number = num_folds)
decayGrid <- expand.grid(decay = seq(0, 0.5, by = 0.05))
model <- train(semrel_f ~ category1f + category2f, data = df[1:cutoff2, ], method = "multinom", trControl = cv, tuneGrid = decayGrid)
decay <- model$bestTune$decay
print(model)}
decay = 0.1
print("Creating models")
y = "semrel_f"
print(colnames(df))
newdf = train %>% select(colnames(train[,c(88:102, 104:118, 124)]))
print("Training baseline model")
bl <- multinom(semrel_f ~ ., MaxNWts = 100000, data =  newdf, decay = decay)
print("Training model 1: multinom model")
actual <- multinom(train[[y]] ~ category1f + category2f, train, MaxNWts = 100000, decay = decay)
print("Training model 2: interactions")
actual2 <- multinom(train[[y]] ~ category1f * category2f, train, MaxNWts = 100000, decay = decay)
if(FALSE){
actual3 <- rpart(train[[y]] ~ category1f + category2f, data = train, method = "class")
actual4 <- nnet(train[[y]] ~ category1f + category2f, train, size = 10, MaxNWts = 100000, maxit = 5000, decay = decay)}
# EVALUATE
names = list("Lexical, multinomial", "Lexical, neural net", "GloVe embeddings, multinomial")
evaluate(list(bl, actual, actual2), y)
if (FALSE){
x = actual
predictions <- predict(x, newdata = test, type = "class")
predictions <- factor(predictions, levels = levels(test[["semrel_f"]]))
conf_matrix <- confusionMatrix(predictions, test[["semrel_f"]])
conf_matrix_df <- as.data.frame(as.table(conf_matrix$table))
colnames(conf_matrix_df) <- c("Actual", "Predicted", "Frequency")
heatmap_plot <- ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "red") +
  geom_text(aes(label = Frequency), vjust = 1) +
  labs(title = "Confusion Matrix Heatmap",
       x = "Actual",
       y = "Predicted") +
  theme_minimal()

print(heatmap_plot)
conf_matrix[["table"]]
conf_matrix[["byClass"]][,c(5, 6, 11)]
accuracy <- conf_matrix$overall["Accuracy"]
print(paste("Accuracy for test set:", accuracy))}