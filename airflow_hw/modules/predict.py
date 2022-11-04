import json
import os
import dill
from datetime import datetime

import pandas as pd


def predict():
    import dill
    path = os.environ.get('PROJECT_PATH', '..')
    name = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{name[0]}', 'rb') as file:
        model = dill.load(file)
    df_prediction = pd.DataFrame(columns=['car_id', 'pred'], index=[0])
    for filename in os.listdir(f'{path}/data/test'):
        with open(os.path.join(f'{path}/data/test', filename), 'r', encoding='utf-8') as fin:
            form = json.load(fin)
            df = pd.DataFrame([form])
            pred = model.predict(df)
            x = {'car_id': df.id, 'pred': pred}
            df_for_join = pd.DataFrame(x)
            df_prediction = pd.concat([df_prediction, df_for_join], axis=0)
    df_prediction.to_csv(f'{path}/data/predictions/prediction_{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    pass


if __name__ == '__main__':
    predict()
