import pandas as pd

def getExpressionData(path, sep=','):
    df = pd.read_csv(path, header=0, sep=sep)
    df = df[df['Slide_no']==1]
    index_df = df[['ROI', 'Segment_type', 'Patient_ID']]
    index_df['Patient_ID'] = index_df['Patient_ID'].apply(lambda x: int(x.split('P')[1])).values
    expression_df = df.iloc[:,-49:]
    df = pd.concat([index_df, expression_df], axis=1)
    df.to_csv(path.split('.')[0]+'.csv', sep=',', header=True, index=False, )

getExpressionData('data/raw/OC1_all.txt', sep='\t')