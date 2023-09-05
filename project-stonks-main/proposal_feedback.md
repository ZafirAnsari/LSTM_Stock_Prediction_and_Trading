Score: 5/8

You’re definitely on the right path, but I’m hoping you can provide a few more specifics on some of your goals. You have a week to revise and resubmit this proposal, but feel free to reach out if you have questions or want advice.

A quick note about data: Yahoo and Kaggle are fine, but you might want to consider using WRDS. To get access, follow these instructions:

> First, you need to create an account here: https://wrds-www.wharton.upenn.edu/register/. You should use your Northwestern email and then indicate that you’re at Northwestern (you can say you’re affiliated with Kellogg, since I think they handle data access for the entire university); that will forward your registration to the folks who handle the subscription, and they should approve your request.
 
> Second, you can use this guide to figure out how to download the data from the WRDS request form: https://github.com/AI4Finance-Foundation/FinRL-Trading/blob/66ba2d906a8e2c2bb0039f2d1872ba80c9e0a0a4/data_processor/README.md
 
> You must agree to use this data for educational purposes only (e.g., you’re not allowed to actually start trading based on your model until you get a commercial license for this or a similar dataset).

Can you provide more basic details about how you plan to (initially) preprocess your data? You don’t need to answer all these questions, but these should guide your answers:
-	How will you split the data into training, validation, and test data? Which years and which stocks fall into which splits?
-	What are you specifically trying to predict? Hourly stock prices? Daily open price? Close price? Close minus open? Weekly trends? Monthly trends?
-	How many timesteps will you initially feed into the LSTM/CNN? A week at a time? The entire timespan of the training set? An LSTM can easily handle a variable-length input, but a CNN can’t as easily.

Are you (just) trying to predict the stock price or are you trying to measure a trading strategy’s return? Your loss function will depend on this. Feel to choose just one for your essential goal and use others in Desired and/or Stretch goals.
-	Binary prediction (e.g., is close greater than open) -> Binary cross entropy loss
-	Estimate price -> MSE loss
-	Actual strategy for buying and selling -> Reinforcement learning with profits/loss as rewards/punishments.

We discussed this in a bit of detail, where you mentioned you want to map from model’s confidence to a “strong buy” to “hold” to “strong sell” ranking which you could incorporate into a trading strategy. We also talked about how if you wanted to use an RL approach as a stretch goal, that you could consider training this in an “end-to-end” manner.

I would avoid framing your desired/stretch goals in terms of accuracy. Obviously it’s implicitly a goal that you want your model to be good, but it’s not one that we want to hold you to in terms of your grade, since it’s not entirely in your control.

What hyperparameters will you consider for your LSTM and CNN? Why? This might change as you learn more about your models, but spend a little while reading through the papers you cite and see which hyperparameters they talk the most about.

We discussed your proposed stretch goal for using CUDA to accelerate your model. It’s fine to leave this as is for now, but by the time you turn in your update it might be helpful to have a bit more specifics about what you might want to do. My main worries are that (a) the course staff might not be able to help you with cuda programming, and (b) not all group members might be able to contribute to that stretch goal. I’m happy to have those worries assuaged, but I’d want to chat a bit more about what you’re thinking.
