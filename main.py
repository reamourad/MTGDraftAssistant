import app.DraftData as dd
import app.ModelBuilder as mb

DATA_PATH = "test.csv"

draft_data = dd.DraftData(DATA_PATH)

pack = draft_data.boosterCreater()
model_builder = mb.ModelBuilder(draft_data)
model_builder.train_model(1)
deck = []
model_builder.predict(deck, pack)

print(len(pack))

