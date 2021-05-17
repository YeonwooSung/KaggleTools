def agg_numeric(df, group_var, df_name):
    '''
    데이터프레임안의 수치데이터에 한하여 대표값들을 계산합니다. 이것은 새로운 feature들을 만드는 데 활용될 수 있습니다.
    (Aggregates the numeric values in a dataframe. This can be used to create features for each instance
     of the grouping variable.)
    
    
    파라미터(Parameters)
    -------------------
        df (dataframe): 연산의 대상이되는 데이터프레임 (the dataframe to calculate the statistics on)
        group_var (string): 그룹화(groupby)의 기준이되는 column (the variable by which to group df)
        df_name (string): column명을 재정의하는데 쓰이는 변수 (the variable used to rename the columns)
        
        
    출력값(Returns)
    -------------------
        agg (dataframe): 
            모든 수치데이터 column들의 대표값들이 연산된 데이터프레임. 각각의 그룹화된 인스턴스들은 대표값(평균, 최소값, 최대값, 합계)
            들을 가짐. 또한, 새롭게 생성된 feature들을 구분하기위해 column들의 이름을 재정의    
            
        agg (dataframe):    
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    '''
    
    # 그룹화 대상이 아닌 id들을 제거
    # Remove id variables other than qrouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns =col)
        
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    
    # 특정 변수들을 그룹화하고 대표값들을 계산
    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count','mean','max','min','sum']).reset_index()
    
    # 새로운 column 이름들을 생성
    # Need to create new column names
    columns = [group_var]
    
    # 변수(원본 데이터프레임의 column name)에 따라 반복문을 실행
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        
        # id column은 생략
        # skip the grouping variable
        if var != group_var:
            
            # 대표값의 종류에 따라 반복문을 생성
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
        
                # 변수 및 대표값의 종류에 따라 새로운 column name을 생성
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
                
    agg.columns = columns
    return agg
