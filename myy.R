getwd()
setwd("C:/Users/crewt/Desktop/college/JALLOSH/OneDrive/Documents/datasets")
# Pract 1
x <- read.csv("ToyotaCorolla.csv")
View(x)

# Select numeric columns of interest
selected_data <- x[, c("price", "age_08_04", "km", "hp", "weight")]

pairs(selected_data, main = "Matrix Plot of Numeric Features", pch = 21, bg = "lightblue")

install.packages("GGally")
library(GGally)
ggpairs(selected_data)

# Use model.matrix to create dummy variables
x$fuel_type <- as.factor(x$fuel_type)
x$met_color <- as.factor(x$met_color)

# Create dummies
dummies <- model.matrix(~ fuel_type + met_color - 1, data = x)

# Combine with the original dataset (excluding original Fuel_Type and Metallic)
data_clean <- cbind(x, dummies)

install.packages("fastDummies")
library(fastDummies)

# Automatically convert all categorical variables to dummies
data_dummy <- dummy_cols(x, select_columns = c("fuel_type", "met_color"), remove_selected_columns = TRUE)
View(data_dummy)

# Pract 2
library(ggplot2)
library(dplyr)
library(lubridate)
library(plotly)

install.packages("plotly")
# Load the data
data <- read.csv("ApplianceShipments.csv")

View(data)

# Convert quarter-year into proper date format
data <- data %>%
  mutate(
    Year = as.numeric(sub(".*-", "", Quarter)),
    Q = sub("-.*", "", Quarter),
    Month = case_when(
      Q == "Q1" ~ 1,
      Q == "Q2" ~ 4,
      Q == "Q3" ~ 7,
      Q == "Q4" ~ 10
    ),
    Date = as.Date(paste(Year, Month, "01", sep = "-"))
  )
View(data)
# Time series line plot
ggplot(data, aes(x = Date, y = Shipments)) +
  geom_line(color = "lightblue", size = 1) +
  labs(title = "Quarterly Appliance Shipments (1985–1989)",
       x = "Year", y = "Shipments (Million $)") +
  theme_minimal()

# ii. Zoom in to 3500–5000 on the Y-Axis
ggplot(data, aes(x = Date, y = Shipments)) +
  geom_line(color = "darkred", size = 1) +
  scale_y_continuous(limits = c(3500, 5000)) +
  labs(title = "Zoomed Quarterly Shipments (3500–5000)", y = "Shipments") +
  theme_minimal()

# iii. Separate Lines for Q1, Q2, Q3, Q4
# Separate lines by quarter
ggplot(data, aes(x = Year, y = Shipments, color = Q, group = Q)) +
  geom_line(size = 1.2) +
  scale_y_continuous(limits = c(3500, 5000)) +
  labs(title = "Quarter-wise Appliance Shipments", y = "Shipments (Million $)", x = "Year") +
  theme_minimal()

# Aggregate by year
yearly_data <- data %>%
  group_by(Year) %>%
  summarise(Annual_Shipments = sum(Shipments))

ggplot(yearly_data, aes(x = Year, y = Annual_Shipments)) +
  geom_line(color = "darkgreen", size = 1.5) +
  labs(title = "Total Yearly Appliance Shipments", y = "Total Shipments (Million $)") +
  theme_minimal()


# Interactive time series plot
plotly::ggplotly(
  ggplot(data, aes(x = Date, y = Shipments, color = Q)) +
    geom_line(size = 1) +
    labs(title = "Interactive Quarterly Shipments", x = "Date", y = "Shipments") +
    theme_minimal()
)

# Pract 3
library(readxl)

# Load data (assume it's in the first sheet)
toyota <- read_excel("ToyotaCorolla.xls")

# View the structure of the dataset
View(toyota)

# Find all categorical (character or factor) columns
categorical_vars <- sapply(toyota, is.character)
categorical_vars
cat_names <- names(toyota)[categorical_vars]
print(cat_names)

# ii. Relationship Between Categorical and Dummy Variables
#When you convert a categorical variable into dummy variables:
#Each category gets a separate binary (0/1) column.
#Only (N − 1) dummy variables are needed for N categories to avoid multicollinearity.

# iii. How Many Dummy Variables for N Categories?
#Answer:If a categorical variable has N levels, then you need N − 1 dummy variables to represent it.

library(fastDummies)

# Convert selected categorical variables to dummies
toyota_dummies <- dummy_cols(toyota, 
                             select_columns = c("Fuel_Type", "Color", "Met_Color"),
                             remove_selected_columns = TRUE)

# View one example record
head(toyota_dummies[1, ])

# Select only numeric columns
numeric_data <- toyota_dummies[sapply(toyota_dummies, is.numeric)]

# Correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
round(cor_matrix[1:5, 1:5], 2)  # View part of it

library(GGally)

# Select key numeric variables for plotting
subset_data <- toyota[, c("Price", "Age_08_04", "KM", "HP", "Weight")]

# Matrix plot
ggpairs(subset_data)

# pract 9
library(tseries)
library(forecast)
library(ggplot2)

# Load data
walmart <- read.csv("WMT.csv")
View(walmart)
# Assume the columns are: Date and Close
# Convert Date column to proper date format
walmart$Date <- as.Date(walmart$Date)

# Sort by Date
walmart <- walmart[order(walmart$Date), ]

# Plot original time series
ggplot(walmart, aes(x = Date, y = Close)) +
  geom_line(color = "blue") +
  labs(title = "Wal-Mart Daily Closing Prices (Feb 2001–Feb 2002)", y = "Close Price ($)", x = "Date") +
  theme_minimal()

# Time series of closing prices
ts_close <- ts(walmart$Close, frequency = 252)  # 252 trading days per year

# First difference of the closing price
DiffClose <- diff(ts_close)


# Plot the differenced series
plot(DiffClose, type = "l", col = "blue",
     main = "Differenced Time Series of Wal-Mart Closing Prices",
     ylab = "Price Change", xlab = "Time")

# Fit AR(1) to the original closing prices
ar1_model <- arima(ts_close, order = c(1, 1, 0))  # AR(1)

# Model summary
summary(ar1_model)

