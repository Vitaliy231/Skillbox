import pandas as pd
import os
import dill
import glob
import json
import logging
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '..')
name = os.listdir(f'{path}/data/models')


def load_model() -> pd.DataFrame:

    with open(f'{path}/data/models/{name[len(name) - 1]}', 'rb') as file:
        model = dill.load(file)
        return model


def model_predict(filename) -> pd.DataFrame:

    with open(filename, 'r', encoding='utf-8') as fin:

        form = json.load(fin)
        df = pd.DataFrame([form])
        model = load_model()
        y = model.predict(df)
        x = {'car_id': df.id, 'pred': y}
        df_for_join = pd.DataFrame(x)
        return df_for_join


def predict() -> None:

    df_prediction = pd.DataFrame()
    for filename in glob.glob(f'{path}/data/test/*.json'):
        df_for_join = model_predict(filename)
        df_prediction = pd.concat([df_prediction, df_for_join], axis=0)

    df_prediction.to_csv(f'{path}/data/predictions/pred_{datetime.now().strftime("%Y%m%d%H%M")}')


if __name__ == '__main__':
    predict()



