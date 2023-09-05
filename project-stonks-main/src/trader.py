import pandas as pd
import matplotlib.pyplot as plt
class Trader():

    def __init__(self,allowShort):
        self.allowShort = allowShort
        self.buyPrice = 0
        self.shortPrice = 0
        self.holdingStock = False
        self.shortingStock = False
        self.currBalance = 0
        self.delta = 0
        self.highestGain = 0
        self.highestLoss = 0
        self.largestSingleGain = 0
        self.largestSingleLoss = 0
        self.longCount = 0
        self.shortCount = 0
        self.initialCost = 0
        self.transactions = [] # Columns are: Initial Price, Final Price, Profit/Loss, Long/Short, Open, Close


    def buyStock(self,price,verbose=False,date = '1-1-2000'): #Enter long position
        if self.shortingStock:
            print("Unable to purchase long buy when shorting")
            return
        if self.holdingStock:
            print("Already holding this security")
            return
        self.buyPrice = price
        self.holdingStock = True
        self.shortingStock = False
        self.transactions.append({"Initial_Price": price, "Long/Short": "Long", "Open" : date})
        if verbose:
            print(f'Buying @ $ {price:.2f}')

    def sellStock(self,price,verbose=False,date = '1-1-2000'): #Exit Long Position
        if self.shortingStock:
            print("Unable to sell long buy while shorting")
        if not self.holdingStock:
            print("Must have stock to sell")
        self.delta = price - self.buyPrice
        self.currBalance += self.delta
        self.holdingStock = False
        self.buyPrice = 0
        self.longCount += 1
        self.transactions[-1].update({"Final_Price": price, "Profit/Loss": self.delta, "Close" : date})
        if verbose:
            print(f'Selling @ $ {price:.2f}')

    def sellShort(self,price,verbose=False,date='1-1-2000'): #Enter Short Position
        if self.holdingStock:
            print("Can''t short while holding long position")
            return
        if self.allowShort:
            self.shortingStock = True
            self.shortPrice = price
            self.transactions.append({"Initial_Price": price, "Long/Short": "Short","Open" : date})
            if verbose:
                print(f'Shorting @ $ {price:.2f}')
        else:
            print("This model is not permitted to short stocks")
        
    def buyBack(self,price,verbose=False,date='1-1-2000'): #Exit Short Position
        if not self.shortingStock:
            print("Can't buy back unshorted position")
            return
        if self.allowShort:
            self.delta = self.shortPrice - price
            self.currBalance += self.delta
            self.shortPrice = 0
            self.shortingStock = False
            self.shortCount += 1
            self.transactions[-1].update({"Final_Price": price, "Profit/Loss": self.delta,"Close" : date})
            if verbose:
                print(f'Closing short @ ${price:.2f}')
        else:
            print("This model is not permitted ot short stocks")
    
    def actOnPrediction(self,todayPrice,predictedPrice,verbose=False,date='1-1-2000'): 
        priceDiff = predictedPrice - todayPrice
        if priceDiff > 0: #Predicted to go higher
            if self.holdingStock: #If already own it,
                return #hold
            if self.shortingStock:
                self.buyBack(todayPrice,verbose=verbose,date=date) #If we forsee upwards movement, we should exit a short position
                self.evaluate()
                return
            #If we have no stake at the moment, buy long
            self.buyStock(todayPrice,verbose=verbose,date=date)
            return
        else: #Predict downward motion
            if self.holdingStock: #If already own it, sell to avoid loss
                self.sellStock(todayPrice,verbose=verbose,date=date)
                self.evaluate()
                return
            if self.shortingStock:#If shorting hold for lower
                return
            if self.allowShort: #If allowed to short and not in any position, enter short
                self.sellShort(todayPrice,verbose=verbose,date=date)
                self.evaluate()
                return
            if verbose:
                print("Predicting Low, Unable to short")
                return
    
    def evaluate(self): #Call for stat tracking
        if self.currBalance > 0 and self.highestGain < self.currBalance:
            self.highestGain = self.currBalance
        elif self.highestLoss > self.currBalance:
            self.highestLoss = self.currBalance
        if self.delta > 0 and self.largestSingleGain < self.delta:
            self.largestSingleGain = self.delta
        elif self.largestSingleLoss > self.delta:
            self.largestSingleLoss = self.delta

    def calculateStats(self):
        if len(self.transactions) == 0:
            return 0, 0
        self.transactions = pd.DataFrame(self.transactions).round(2)
        correctPreds = self.transactions.loc[self.transactions['Profit/Loss'] > 0].count()
        avgProfit = self.transactions['Profit/Loss'].mean()

        return correctPreds['Profit/Loss'], avgProfit

    def printStats(self):
        correctPreds, avgProfit = self.calculateStats()

        print(f"Final Net Balance Change: ${self.currBalance:.2f}")
        print(f"Highest Balance: ${self.highestGain:.2f}")
        print(f"Lowest Balance: ${self.highestLoss:.2f}")
        print(f"Biggest Single Play: ${self.largestSingleGain:.2f}")
        print(f"Worst Single Play: ${self.largestSingleLoss:.2f}\n")
        print(f"With {self.longCount + self.shortCount} total transactions ({correctPreds} profitable)")
        print(f"{self.longCount} Long Positions")
        print(f"{self.shortCount} Short Positions")
        print(f"${avgProfit:.2f} Average Profit per Transaction")
        print(f"{(self.currBalance  /self.initialCost)*100:.2f}% Change from initial")

    def closePositions(self,price,date='01-01-2000'):
        if self.holdingStock:
            self.sellStock(price,date=date)
        if self.shortingStock:
            self.buyBack(price,date=date)
        self.evaluate()
        
    def plotTrades(self,stockFile):
        data = pd.read_csv(stockFile,usecols=['date','close'])
        data['date'] = pd.to_datetime(data['date'])
        for trade in self.transactions:
            color = 'b'
            if trade['Long/Short'] == "Long":
                color = 'g'
            else:
                color = 'r'
            start = pd.to_datetime(trade['Open'])
            end = pd.to_datetime(trade['Close'])
            newdata = data[start<data['date']]
            newdata = newdata[end>data['date']]
            plt.plot(newdata['date'],newdata['close'],color=color)
        plt.title("Trades Visualized")
        plt.xlabel("Dates")
        plt.ylabel("Stock Price")
        plt.show()
