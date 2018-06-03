########################
########################
## Feature Extraction ##
########################
########################


# extracting features from time series of data in spiral drawings data set
def extract_features(df):
    
    XList = df['X'].tolist()
    XValues = [float(i) for i in XList]
    YList = df['Y'].tolist()
    YValues = [float(i) for i in YList]
    ZList = df['Z'].tolist()
    ZValues = [float(i) for i in ZList]
    PressList = df['Pressure'].tolist()
    PressValues = [float(i) for i in PressList]
    GripList = df['GripAngle'].tolist()
    GripValues = [float(i) for i in GripList]
    IDList = df['ID'].tolist()
    IDValues = [float(i) for i in IDList]
    
    ID = IDList[1]
    
    feature1Z = mean(ZValues)
    feature1Y = mean(YValues)
    feature1X = mean(XValues)
    feature1P = mean(PressValues)
    feature1G = mean(GripValues)
    
    feature2X = np.std(XValues, axis=0)
    feature2Y = np.std(YValues, axis=0)
    feature2Z = np.std(ZValues, axis=0)
    feature2P = np.std(PressValues, axis=0)
    feature2G = np.std(GripValues, axis=0)
    
    vZ = [abs(x[1]-x[0]) for x in zip(ZValues[1:], ZValues[:-1])]
    summeZ = sum(vZ)
    vX = [abs(x[1]-x[0]) for x in zip(XValues[1:], XValues[:-1])]
    summeX = sum(vX)
    vY = [abs(x[1]-x[0]) for x in zip(YValues[1:], YValues[:-1])]
    summeY = sum(vY)
    vP = [abs(x[1]-x[0]) for x in zip(PressValues[1:], PressValues[:-1])]
    summeP = sum(vP)
    vG = [abs(x[1]-x[0]) for x in zip(GripValues[1:], GripValues[:-1])]
    summeG = sum(vG)
    
    feature3Z = summeZ/(len(ZValues)-1)
    feature3X = summeX/(len(XValues)-1)
    feature3Y = summeY/(len(YValues)-1)
    feature3P = summeP/(len(PressValues) -1)
    feature3G = summeG/(len(GripValues) -1)
    
    wX = [abs(x[1]-x[0]) for x in zip(XValues[2:],XValues[:-2])]
    summeWX = sum(wX)
    wY = [abs(x[1]-x[0]) for x in zip(YValues[2:],YValues[:-2])]
    summeWY = sum(wY)
    wZ = [abs(x[1]-x[0]) for x in zip(ZValues[2:],ZValues[:-2])]
    summeWZ = sum(wZ)
    wP = [abs(x[1]-x[0]) for x in zip(PressValues[2:],PressValues[:-2])]
    summeWP = sum(wP)
    wG = [abs(x[1]-x[0]) for x in zip(GripValues[2:],GripValues[:-2])]
    summeWG = sum(wG)
    
    feature4X = summeWX/(len(XValues)-1)
    feature4Y = summeWY/(len(YValues)-1)
    feature4Z = summeWZ/(len(ZValues)-1)
    feature4P = summeWP/(len(PressValues)-1)
    feature4G = summeWG/(len(GripValues)-1)

    feature5X = (feature3X/feature2X)
    feature5Y = (feature3Y/feature2Y)
    feature5Z = (feature3Z/feature2Z)
    feature5P = (feature3P/feature2P)
    feature5G = (feature4G/feature2G)

    feature6X = (feature5X/feature2X)
    feature6Y = (feature5Y/feature2Y)
    feature6Z = (feature5Z/feature2Z)
    feature6P = (feature5P/feature2P)
    feature6G = (feature5G/feature2G)
    
    feature7Z = median(ZValues)
    feature7Y = median(YValues)
    feature7X = median(XValues)
    feature7P = median(PressValues)
    feature7G = median(GripValues)

    feature8Z = variance(ZValues)
    feature8Y = variance(YValues)
    feature8X = variance(XValues)
    feature8P = variance(PressValues)
    feature8G = variance(GripValues)
    
    list1 = [feature1X,feature1Y,feature1Z, feature1P,feature1G,
         feature2X,feature2Y,feature2Z, feature2P,feature2G,
         feature3X,feature3Y,feature3Z, feature3P,feature3G,
         feature4X,feature4Y,feature4Z, feature4P,feature4G,
         feature5X,feature5Y,feature5Z, feature5P,feature5G,
         feature6X,feature6Y,feature6Z, feature6P,feature6G,
         feature7X,feature7Y,feature7Z, feature7P,feature7G,
         feature8X,feature8Y,feature8Z, feature8P,feature8G,
         df.shape[0],
         ID]
    
    return list1


row0 = extract_features(df0)
row1 = extract_features(df1)
row2 = extract_features(df2)
row3 = extract_features(df3)
row4 = extract_features(df4)
row5 = extract_features(df5)
row6 = extract_features(df6)
row7 = extract_features(df7)
row8 = extract_features(df8)
row9 = extract_features(df9)
row10 = extract_features(df10)
row11 = extract_features(df11)
row12 = extract_features(df12)
row13 = extract_features(df13)
row14 = extract_features(df14)
row15 = extract_features(df15)
row16 = extract_features(df16)
row17 = extract_features(df17)
row18 = extract_features(df18)
row19 = extract_features(df19)
row20 = extract_features(df20)
row21 = extract_features(df21)
row22 = extract_features(df22)
row23 = extract_features(df23)
row24 = extract_features(df24)

dataframe = pd.DataFrame([row0,row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12,row13,row14,row15,row16,row17,row18,row19,row20,row21,row22,row23,row24],
                         columns=['feature1X','feature1Y','feature1Z','feature1P','feature1G','feature2X','feature2Y','feature2Z', 'feature2P','feature2G',
         'feature3X','feature3Y','feature3Z', 'feature3P','feature3G',
         'feature4X','feature4Y','feature4Z', 'feature4P','feature4G',
         'feature5X','feature5Y','feature5Z', 'feature5P','feature5G',
         'feature6X','feature6Y','feature6Z', 'feature6P','feature6G',
         'feature7X','feature7Y','feature7Z', 'feature7P','feature7G',
         'feature8X','feature8Y','feature8Z', 'feature8P','feature8G',
         'length',
         'ID'])
