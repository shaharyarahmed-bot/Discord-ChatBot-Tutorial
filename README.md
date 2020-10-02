# Discord-ChatBot
Hey everyone this Github repository is for the Discord Chatbot I made. Please continue reading if you want your chatbot to work properly! Before starting off I would like to say Thank you to Tech With Tim as his tutorials were really helpful!

# Installation
In order to make this chatbot you need to install all the packages provided in the requirements.txt and also make sure you have Python 3.7 installed. In your terminal (obviously after installing the repository) go into the repositories directory and type in:

pip install -r requirements.txt if the first one does not work then:  pip3 install -r requirements.txt
then type in:

pip install -U discord.py[voice]

# Setting up our Discord Sever and Application
First of all we need to setup our discord server, so go ahead and create a new discord server. Then after creating a new server go to the Discord developer portal:
https://discord.com/developers/applications

And create a new application name it whatever you want. After that click on your application and you will see a setting named "Bot" go ahead and click on that. Over there create a new bot and specify its previllages (I made it administrator). 

Then ahead to: https://discord.com/oauth2/authorize?client_id=CLIENT_ID&scope=bot&permissions=0 and where it says CLIENT_ID (in the link) replace it with your bots client id. After doing that add the Bot to the Discord server you just made.

After all that copy your bot's token and paste it in the tokens.txt file in the repository (Make sure you dont give this to anyone!)

# Adding AI To Our Chatbot
In order to give our discord bot life we first need to train it. So go ahead in the Bot folder in the repository and then in the model folder and adjust the intents.json file as you wish. I have added a few basic tags you can add or delete new tags but make sure its in the correct format!

After adjusting the intents.json go back into the Bot folder and inside bot.py. at the end of the program add Chatbot.train(epochs = epoch number) instead of epoch number type in a actual integer. If you want to learn what a epoch number is then search it up! the default is 5000 but you will have to adjust it according to your intents.json file. And then run main.py and watch it train the Chatbot! Make sure you read the accuracy at the end of the training in the terminal and make sure it meets your requirement. Mine was 99%

# Running the Discord Chatbot
After completing all the steps above. Get out of the Chatbot folder and you will see another main.py file (yeah I know I am not good at naming files) before running the main.py file make sure you remove Chatbot.train() from your main.py file in the Chatbot folder or else it will start training your bot again!

After removing that line run the file and voila! Your chatbot should start working!

If you made it this far then congratulations! Your chatbot should be working if you have any errors please let me know at my email:
shaharyar.ahmed1121@gmail.com
