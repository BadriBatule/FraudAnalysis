# ---------------------------
# Required Libraries
# ---------------------------

library(readr)
library(dplyr)
library(caret)
library(neuralnet)
library(pROC)
library(PRROC)
library(shiny)

# ---------------------------
# Load Dataset
# ---------------------------

creditcard <- read_csv("Downloads/creditcard.csv")

# ---------------------------
# Reusable Function: run_fraud_model()
# ---------------------------

run_fraud_model <- function(data) {
  # Normalize Time and Amount
  data$Time <- (data$Time - min(data$Time)) / (max(data$Time) - min(data$Time))
  data$Amount <- (data$Amount - min(data$Amount)) / (max(data$Amount) - min(data$Amount))
  
  # Undersample to balance classes
  fraud <- data %>% filter(Class == 1)
  non_fraud <- data %>% filter(Class == 0)
  set.seed(42)
  non_fraud_sample <- non_fraud %>% sample_n(nrow(fraud))
  balanced_data <- bind_rows(fraud, non_fraud_sample) %>% sample_frac(1)
  
  # Train/Test Split
  split_index <- createDataPartition(balanced_data$Class, p = 0.7, list = FALSE)
  train_data <- balanced_data[split_index, ]
  test_data <- balanced_data[-split_index, ]
  
  # Neural Network (Shallow)
  features <- names(train_data)[names(train_data) != "Class"]
  formula_nn <- as.formula(paste("Class ~", paste(features, collapse = " + ")))
  set.seed(123)
  nn_model <- neuralnet(formula_nn, data = train_data, hidden = c(5), linear.output = FALSE)
  
  # Neural Network (Deep)
  nn_model_deep <- neuralnet(formula_nn, data = train_data, hidden = c(10, 5), linear.output = FALSE)
  
  # Predictions (Shallow)
  test_features <- subset(test_data, select = -Class)
  nn_predictions <- compute(nn_model, test_features)$net.result
  predicted_class <- ifelse(nn_predictions > 0.5, 1, 0)
  
  # Evaluation
  cm <- confusionMatrix(as.factor(predicted_class), as.factor(test_data$Class))
  roc_obj <- roc(test_data$Class, as.numeric(nn_predictions))
  auc_score <- auc(roc_obj)
  scores <- as.numeric(nn_predictions)
  pr <- pr.curve(scores.class0 = scores[test_data$Class == 1],
                 scores.class1 = scores[test_data$Class == 0], curve = TRUE)
  pr_auc <- pr$auc.integral
  
  # Summary
  summary_df <- data.frame(
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    ROC_AUC = auc_score,
    PR_AUC = pr_auc
  )
  
  return(list(
    model = nn_model,
    model_deep = nn_model_deep,
    train_features = features,
    scaler = list(Time = range(data$Time), Amount = range(data$Amount)),
    metrics = summary_df,
    roc = roc_obj,
    pr = pr
  ))
}

# ---------------------------
# Run and Export Outputs
# ---------------------------

results <- run_fraud_model(creditcard)

write.csv(results$metrics, "fraud_model_summary.csv", row.names = FALSE)

pdf("fraud_network_plot.pdf")
plot(results$model)
dev.off()

pdf("fraud_deep_network_plot.pdf")
plot(results$model_deep)
dev.off()

# ---------------------------
# Shiny App with Deep NN + Upload
# ---------------------------

ui <- fluidPage(
  titlePanel("Fraud Detection Neural Network"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV (Same format, no 'Class')", accept = ".csv"),
      actionButton("predict", "Run Prediction on Uploaded Data")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Shallow Network", plotOutput("nn_plot")),
        tabPanel("Deep Network", plotOutput("deep_nn_plot")),
        tabPanel("ROC Curve", plotOutput("roc_plot")),
        tabPanel("Metrics", tableOutput("metrics_table")),
        tabPanel("Uploaded Predictions", tableOutput("uploaded_pred"))
      )
    )
  )
)

server <- function(input, output) {
  data <- creditcard
  result <- run_fraud_model(data)
  
  output$nn_plot <- renderPlot({
    plot(result$model)
  })
  
  output$deep_nn_plot <- renderPlot({
    plot(result$model_deep)
  })
  
  output$roc_plot <- renderPlot({
    plot(result$roc, main = "ROC Curve")
  })
  
  output$metrics_table <- renderTable({
    result$metrics
  })
  
  observeEvent(input$predict, {
    req(input$file)
    new_data <- read_csv(input$file$datapath)
    
    # Normalize Time and Amount
    time_min <- result$scaler$Time[1]
    time_max <- result$scaler$Time[2]
    amt_min <- result$scaler$Amount[1]
    amt_max <- result$scaler$Amount[2]
    new_data$Time <- (new_data$Time - time_min) / (time_max - time_min)
    new_data$Amount <- (new_data$Amount - amt_min) / (amt_max - amt_min)
    
    # Ensure feature alignment
    new_data <- new_data[, result$train_features]
    
    # Predict with shallow model
    preds <- compute(result$model, new_data)$net.result
    pred_classes <- ifelse(preds > 0.5, 1, 0)
    
    pred_df <- new_data
    pred_df$Fraud_Probability <- round(preds, 4)
    pred_df$Predicted_Class <- pred_classes
    
    output$uploaded_pred <- renderTable({
      head(pred_df, 20)
    })
  })
}

shinyApp(ui = ui, server = server)

