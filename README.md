# NoBullying

Authors: Aryan Gorwade, Ayushman Chakraborty, Jasamarbir Arora, Pramukh Bhushan

This discord bot stops all form of racism and harmful language in a Discord server's channel. It uses machine 
learning to detect bullying and racism in a chat and steers the conversation away from the bullying after giving 
warnings to the people involved by providing relevant conversation starters. 

The bot was based on the idea that simply banning or kicking out members was not enough since they could always join back with an alternate account and likely would not learn much from this. Slowly impacting them though, through small ways like this, can help to change abusive and prejudiced members' habits. 

The bot combines a vectorized logistic regression model to detect bullying with a natural language processing model
that is able to find a related topic and generate a conversation starter about that topic to divert the conversation
away from hurtful language. The logistic regression model is trained on the training data only once, while the bot is being initialized, before the 
resulting model is used on all Discord conversations after that. 

It makes use of several libraries, including the following: nltk, sklearn, matplotlib, gensim, dotenv, pandas, etc.

For more information, please check out https://devpost.com/software/racism-stopper and the video linked in the page. This bot 
was created for the 2022 Impact hacks hackathon.
