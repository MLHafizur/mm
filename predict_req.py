import requests
import time
import pandas as pd
from mlserver.codecs.pandas import PandasCodec # required mlserver>=1.1.0, 
payloads = "./IDA_en.ndjson"

start = time.time()
df = pd.read_json(payloads, lines=True)

df = df.fillna('')


payload = PandasCodec.encode_request(df, use_bytes=False)



response = requests.post("http://10.0.0.139:8008/v2/models/multiclass-En/infer", json=payload.dict())
end = time.time()
print(response.text)
print("Time taken: ", end - start)
