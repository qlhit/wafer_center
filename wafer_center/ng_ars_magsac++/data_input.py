import pandas as pd

class data_input:
    def get_data(self, path):
        df = pd.read_excel('data.xlsx', sheet_name='Sheet1', header=0)  # 读取指定工作表
        filtered_data = df[df['价格'] > 20]   