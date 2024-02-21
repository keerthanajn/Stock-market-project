import yfinance as yf
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500.index
sp500.plot.line(y="Close", use_index=True)
if "Dividends" in sp500.keys():
    del sp500["Dividends"]
if "Stock Splits" in sp500.keys():
    del sp500["Stock Splits"]
sp500["Tomorrow"]=sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state =1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
from sklearn.metrics import precision_score
pred = model.predict(test[predictors])
import pandas as pd
pred = pd.Series(pred, index = test.index)
precision_score(test["Target"], pred)
combined = pd.concat([test["Target"],pred], axis=1)
combined.plot()

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    pred = model.predict(test[predictors])
    pred = pd.Series(pred, index = test.index, name="Predictions")
    combined = pd.concat([test["Target"],pred], axis=1)
    return combined

predictions = backtest(sp500, model, predictors)
