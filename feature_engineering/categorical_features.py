def count_categorical(df, group_var, df_name):
    """
    group_var에 대해 각 고유 범주들의 counts 및 normalized counts를 계산  
    
    Computes counts and normalized counts for each observation of `group_var` of each unique category
    in every categorical variable
    
    
    파라미터 (Parameters)
    ---------------------
    
    df (dataframe): 연산의 대상이되는 데이터프레임 (the dataframe to calculate the statistics on)
    group_var (string): 그룹화(groupby)의 기준이되는 column (the variable by which to group df)
    df_name (string): column명을 재정의하는데 쓰이는 변수 (the variable used to rename the columns)
    
    
    출력값(Returns)
    -------------------
    categorical : 데이터프레임
     group_var에 대해 각 고유 범주들의 counts 및 normalized counts의 값이 포함된 데이터프레임  
    
    categorical : dataframe
    A dataframe with counts and normalized counts of each unique category in every categorical variable
    with one row for every unique value of the `group_var
    
    """
    # 범주형 column들을 선택
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    
    # 확실히 id가 column에 있도록 지정 
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    
    # group_var를 기준으로 그룹화하고 sum과 mean을 계산
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum','mean'])
    
    column_names = []
    
    # level 0의 column들에 따라 반복문을 실행
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        
        # level 1의 통계값들에 대해 반복문을 실행
        # Iterate through the stats in level 1
        for stat in ['count','count_norm']:
            column_names.append('%s_%s_%s' % (df_name, var, stat)) 
    
    
    categorical.columns = column_names
    
    return categorical
