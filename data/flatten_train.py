"""
Converts input csv file of type: img_name, bbox, List[queries]
to output csv file of type: img_name, bbox, query
"""
import fire
import pandas as pd
from tqdm import tqdm
import copy
import ast


def converter(inp_csv, out_csv):
    inp_df = pd.read_csv(inp_csv)
    inp_df['query'] = inp_df['query'].apply(
        lambda x: ast.literal_eval(x))

    inp_df = inp_df.to_dict(orient='records')
    out_list = []
    for row in tqdm(inp_df):
        queries = row.pop('query')
        for query in queries:
            out_dict = copy.deepcopy(row)
            out_dict['query'] = query
            out_list.append(out_dict)

    out_df = pd.DataFrame(out_list)
    out_df.to_csv(out_csv, index=False, header=True)


if __name__ == '__main__':
    fire.Fire(converter)
