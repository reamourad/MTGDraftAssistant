# MLFromScratch

AI agent to predict optimal card picks during a Magic: The Gathering (MTG) Draft based on real player data from 17Lands. Model made to follow high-skilled player choices with win rate over 60%

MTG drafter using Keras and Pandas, data for the draft picking comes from 17Lands 

v0: Made a simple LLM that can make a 45 card deck based on a given MTG set 

v1: Upgraded the learning model by tweaking the value of our optimizer Adam, the size of the LSTM layer and the epochs 

v1.5: Added the rules of drafting in MTG, the model now can act as an user in a drafting environment

v2: Upgraded to a Transformer-based sequence model

Download python 3.10

windows:
python3.10 -m venv venv      # create virtual environment
venv\Scripts\activate   # activate it
pip install -r requirements.txt

mac: 
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Data Preparation
The model is trained on draft data sourced from 17Lands.

Download Data: Download the relevant set data in .csv.gz format from 17Lands.

Process Data: Change the name of the .csv.gz in the script. Run the preparation script to take only 1 000 000 lines from the dataset.
python ./app/prepare_dataset.py
