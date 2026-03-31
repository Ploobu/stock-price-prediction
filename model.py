# Importing libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Downloading Apple Stock Data
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
df.columns = ['_'.join(col).strip() for col in df.columns]
df = df[['Close_AAPL', 'Volume_AAPL']] # Only 2 columns: the closing price (price at end of each day) and volume (how many shares were traded)
df = df.rename(columns={'Close_AAPL': 'Close', 'Volume_AAPL': 'Volume'})
print(df.head()) # checking it looks right


# Moving averages - the average closing price over the last 10 and 50 days
# Create new columns:
df['MA_10'] = df['Close'].rolling(10).mean()
df['MA_50'] = df['Close'].rolling(50).mean()

# Daily return - how much the price changed today as a percentage
df['Return'] = df['Close'].pct_change()

# Volatility - how much the returns have been jumping around over the last 10 days
# High volatility measn the price is movin a lot, low means stable
df['Volatility'] = df['Return'].rolling(10).std()

# Target variable: 1 = price goes up tomorrow, 0 = goes down
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # what we are trying to predict 

# Remove rows with missing values
df.dropna(inplace=True)

print(df.head())

# Training the model:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Separating data into X (the inputs - the features engineered) and y (the output - what we want to predict)
features = ['MA_10', 'MA_50', 'Return', 'Volatility', 'Volume']
X = df[features]
y = df['Target']

# Split into 2 halves - 80% model will learn from (training) and 20% for testing (pretned model hasn't seen this and check how well it predicts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False) # shuffle is like so bc this is time series data - has to stay in chronological order

# Create and train a random forest - it is an algorithm that builds lots of decision trees and combines their prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model will now predict data it hasn't seen and then compare them to the actual answers.
# The accuracy score will tell us the % it gets right
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print(classification_report(y_test, preds))


# Visualising results
importances = pd.Series(model.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', title='Feature Importance', color='steelblue')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()