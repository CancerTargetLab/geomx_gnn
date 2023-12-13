import pandas as pd

def getExpressionData(path, sep=','):
    df = pd.read_csv(path, header=0, sep=sep)
    df = df[df['Slide_no']==1]
    index_df = df[['ROI', 'Segment_type', 'Patient_ID']]
    index_df['Patient_ID'] = index_df['Patient_ID'].apply(lambda x: int(x.split('P')[1])).values
    expression_df = df.iloc[:,-49:]
    df = pd.concat([index_df, expression_df], axis=1)
    df.to_csv(path.split('.')[0]+'.csv', sep=',', header=True, index=False, )

def getAllPositionData(path, sep=','):
    df = pd.read_csv(path[0], header=0, sep=sep)
    df = df[['Image', 'Centroid.X.px', 'Centroid.Y.px', 'Class']]
    for p in path[1:]:
        df1 = pd.read_csv(p, header=0, sep=sep)
        df1 = df[['Image', 'Centroid.X.px', 'Centroid.Y.px']]
        df = pd.concat([df, df1])
    df1 = pd.DataFrame()
    for roi in df['Image'].unique().tolist():
        df2 = df[df['Image']==roi]
        mask = ~df2.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep=False) | ~df2.duplicated(subset=['Centroid.X.px', 'Centroid.Y.px'], keep='first')
        df2 = df2[mask]
        df1 = pd.concat([df1, df2])
    df1.to_csv('data/measurements_w_class.csv', sep=',', header=True, index=False, )

#getExpressionData('data/dataset.csv', sep=',')
getAllPositionData(['data/measurements_CD8_roi.tsv', 'data/measurements_CD45_roi.tsv'], sep='\t')