#Import Libraries; (Some installations may required to be installed before running this code)
import pandas as pd
import numpy as np
import glob as glob
import os
import graphviz
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO    
from sklearn.preprocessing import LabelEncoder
from IPython.display import Image
import matplotlib.pyplot as plt

#Criteria for each Testing Parameters
Criterion = ['entropy', 'gini']
Test1 = {'MaxDepth': range(1,31),'Impurity': 0.0,'MaxLeaf': None,'Name': "Max_Depth",'Table_Name': "Max_Depth",'TwoParameters': False}
Test2 = {'MaxDepth': None,'Impurity': 0.0,'MaxLeaf': range(2,31),'Name': "Max_Leaf_Nodes",'Table_Name': "Max_Leaf_Nodes",'TwoParameters': False}
Test3 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005,.01,.05,.1,.5,1],'MaxLeaf': None,'Name': "Impurity_Decreased",'Table_Name': "Impurity_Decreased",'TwoParameters': False}
Test4 = {'MaxDepth': range(1,31),'Impurity': [.00005,.0001,.0005,.001,.005,.01,.05,.1,.5,1],'MaxLeaf': None,'Name': "Impurity_Decreased_Plus_Max_Depth",'Table_Name': "Max_Depth",'TwoParameters': True}
Test5 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005,.01,.05,.1,.5,1],'MaxLeaf': range(2,31),'Name': "Impurity_Decreased_Plus_Max_Leaf_Nodes",'Table_Name': "Max_Leaf_Nodes",'TwoParameters': True}
Test_List = [Test1,Test2,Test3,Test4,Test5]
#This path is used multiple times, stored first half to make lines shorter
originalPath = "/media/southpark86/AMG1/School/Spring 2020/Intrusion Detection/Individual Project/"

#All UNSW Attributes
def UNSW_DataSet_Parameters():
    #Dictionary of all Attacks in Data Set
    mappingUNSW = {
    "Benign": 0,
    " Fuzzers": 1,
    "Reconnaissance": 2,
    "Shellcode": 3,
    "Analysis": 4,
    "Backdoor": 5,
    "DoS": 6,
    "Exploits": 7,
    "Generic": 8,
    "Worms": 9,
    }
    #Columns to review UNSW data and add to dictionary
    UNSWcols = [0,1,2,3,11,45]
    path = originalPath+"UNSW_Data_Set/"
    data_set_path = path +"UNSWDataSet.csv"
    attack_names_path= path +"UNSWDataSetAttackNames.csv"
    pathD = originalPath+"UNSW-NB15"
    return data_set_path,attack_names_path, pathD, None, None, mappingUNSW, UNSWcols, 44, 45, 10, [1,3], ["Benign"], [47]

#Loads the data set in order to get it ready to use
def Load_Data_Set():
    data_set_path,attack_names_path,pathD,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=UNSW_DataSet_Parameters()

    #Loading Saved CSV    
    #dataset = pd.read_csv(data_set_path, header=header, index_col=indexCol)
    #dataset = dataset.drop(dataset.columns[0],axis=1)

    #attackNames = pd.read_csv(attack_names_path, header=header, index_col=indexCol)
    #attackNames = attackNames.iloc[:,1]
    #attackNames = attackNames.unique()
    #print(list(attackNames).index('Benign'))
    #print(list(attackNames).index('Analysis'))
    print("Pre-processing data...")

    #get formatted pandas dataset
    dataset = formatData(pathD, header, indexCol)

    #Drop columns not using
    if(len(dropFeats) != 0):
        dataset.drop(dropFeats, axis=1, inplace=True)
    #Fill missing data with columns with missing data
    if(len(missReplacement) != 0):
        dataset = missData(dataset, missReplacement, missCols)

    #replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    dataset = dataset.replace(to_replace=-np.inf, value=np.nan)
    #Added This Replace To Help with CIC Dataset
    dataset = dataset.replace(to_replace="Infinity", value=np.nan)

    #drop any row with a NaN
    dataset = dataset.dropna(how="any")
    attackNames = []

    #Replaces attacks with numbers
    for columnL in colL:
        encodeData = LabelEncoder()
        #saves attack names to use it later to name each node class
        if columnL == labelCol:
            attackNames = dataset.iloc[:,columnL]
        dataset.iloc[:,columnL] = encodeData.fit_transform(dataset.iloc[:,columnL])
    
    attackNames = attackNames.unique()
    return dataset,list(attackNames),labelCol

#Formats all csv files into one Dataframe
def formatData(path, head, indexCol):
    os.chdir(path)
    fileList = glob.glob("*.csv")
    dataList = []

    for file in fileList:
        data = pd.read_csv(file, header=head, index_col=indexCol)
        dataList.append(data)
    dataset = pd.concat(dataList, axis=0)

    return dataset

#takes a dataset, list of value to replace missing data in corresponding column in missCols
def missData(dataset, miss, missCols):
    i = 0
    for col in missCols:
        dataset[col].fillna(value=miss[i], inplace=True)
        i+=1

    return dataset


#   DECISION TREE STARTS HERE #
#Splits Data into training and testing, Returns splitted data
def trainModels(dataset,labelCol):    
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    y = dataset.iloc[:,labelCol]
    X = dataset.drop(dataset.columns[labelCol],axis=1)

    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)

    return X,y,X_train,y_train,X_test,y_test

#Creates Tree Model and Returns predicted Accuracy
def CreateModel(Criterion,Impurity,MaxDepth,MaxLeaf,Test_Data,X,X_train,y_train,X_test,y_test,name,attackNames):
    print("Creating Models")
    model = DecisionTreeClassifier(criterion=Criterion, min_impurity_decrease=Impurity,
                                    max_depth= MaxDepth, max_leaf_nodes=MaxLeaf)
    
    #Training the decision tree classifier 
    model.fit(X_train,y_train)
    print("Training Decision Tree Complete")

    #print specific trees, because png files cannot store too much data without sizing it down
    if (Test_Data['MaxDepth']!= None ):
        if MaxDepth<11:
            printModels(model=model,X=X,name=name,attackNames=attackNames)
    #    print('Nothig Printed')

    #Predicting test Accuracies
    pred =  model.predict(X_test)
    pred_acc= accuracy_score(y_test, pred)

    return pred_acc

#Creates Creates and Saves a .png Image to selected Folder
def printModels(model,X,name,attackNames):
    print("Printing Models")
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X.columns,  
                      class_names=attackNames, 
                      filled=True, rounded=True,
                      special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(name)
    Image(graph.create_png())

#   RUNNING DECISION TREE
def Run_DecisionTree(dataset,labelCol,attackNames,Test_Data,Criterion):
    #Split and Train Data
    X,y,X_train,y_train,X_test,y_test= trainModels(dataset,labelCol)

    #use to plot gini vs entropy accuracies
    x_label = []
    acc_gini = []
    acc_entropy = []

    #Create and Printing Models, Appends predicted accuracies based on Criterion, Appends MaxDepth
    print("STARTIMG LOOP")
    #Checks Whether or not it's Testing for Two Parameters
    if(Test_Data['TwoParameters']):
        print("TWO PARAMETERS")
        LoopThis = Test_Data['MaxDepth'] if (Test_Data['MaxDepth'] != None) else Test_Data['MaxLeaf']
        for ImpureLoop in Test_Data['Impurity']:
            x_label = []
            acc_gini = []
            acc_entropy = []
            for LoopParameter in LoopThis:
                for Crit in Criterion :
                    path= originalPath+"DecisionTreeIDS/DecisionTreeResults/Criterion_Gini/Impurity_Decreased/Impurity_"+str(ImpureLoop)+"/" if (Crit == 'gini')  else originalPath+"DecisionTreeIDS/DecisionTreeResults/Criterion_Entropy/Impurity_Decreased/Impurity_"+str(ImpureLoop)+"/"
                    name = 'UNSW_Dataset_Features_Criterion_Gini_'+Test_Data['Name']+'_'+str(LoopParameter)+'_Impurity_'+str(ImpureLoop)+'.png' if (Crit == 'gini') else 'UNSW_Dataset_Features_Criterion_Entropy_'+Test_Data['Name']+'_'+str(LoopParameter)+'_Impurity_'+str(ImpureLoop)+'.png'
                    
                    MaxDepth = LoopParameter if Test_Data['MaxDepth'] != None else Test_Data['MaxDepth']
                    Impurity = ImpureLoop
                    MaxLeaf = LoopParameter if Test_Data['MaxLeaf'] != None else Test_Data['MaxLeaf']

                    name = path+name
                    #Creates Gini Models
                    pred_acc= CreateModel(Criterion=Crit,Impurity=Impurity,MaxDepth=MaxDepth,MaxLeaf=MaxLeaf,Test_Data=Test_Data, X=X,
                                                                X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                                name=name, attackNames=attackNames)
                    if (Crit == 'gini') :
                        acc_gini.append(pred_acc)
                    else:
                        acc_entropy.append(pred_acc)
                x_label.append(LoopParameter)
                print("Loop: "+str(LoopParameter))

            plottingGraphs(x_label=x_label,acc_gini=acc_gini,acc_entropy=acc_entropy,Test_Data=Test_Data,ImpureLoop=ImpureLoop)
    else:
        print("ONE PARAMETER") 
        LoopThis = Test_Data['MaxDepth'] if (Test_Data['MaxDepth'] != None) else (Test_Data['MaxLeaf'] if (Test_Data['MaxLeaf'] != None) else Test_Data['Impurity'])
        for LoopParameter in LoopThis:
            for Crit in Criterion :
                path= originalPath+"DecisionTreeIDS/DecisionTreeResults/Criterion_Gini/" if (Crit == 'gini')  else originalPath+"DecisionTreeIDS/DecisionTreeResults/Criterion_Entropy/"
                name = 'UNSW_Dataset_Features_Criterion_Gini_'+Test_Data['Name']+'_'+str(LoopParameter)+'.png' if (Crit == 'gini')  else 'UNSW_Dataset_Features_Criterion_Entropy_'+Test_Data['Name']+'_'+str(LoopParameter)+'.png'
                
                MaxDepth = LoopParameter if (Test_Data['MaxDepth'] != None) else Test_Data['MaxDepth']
                Impurity = LoopParameter if (Test_Data['Impurity'] != 0.0) else Test_Data['Impurity']
                MaxLeaf = LoopParameter if (Test_Data['MaxLeaf'] != None) else Test_Data['MaxLeaf']

                name = path+name
                #Creates Gini Models
                pred_acc= CreateModel(Criterion=Crit,Impurity=Impurity,MaxDepth=MaxDepth,MaxLeaf=MaxLeaf,Test_Data=Test_Data, X=X,
                                                            X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                            name=name, attackNames=attackNames)
                if (Crit == 'gini') :
                    acc_gini.append(pred_acc)
                else:
                    acc_entropy.append(pred_acc)
            x_label.append(LoopParameter)
            print("Loop: "+str(LoopParameter))

        plottingGraphs(x_label=x_label,acc_gini=acc_gini,acc_entropy=acc_entropy,Test_Data=Test_Data,ImpureLoop=0)

def plottingGraphs(x_label,acc_gini,acc_entropy,Test_Data,ImpureLoop):
    #stores the both gini and entropy accuraccies into a dataframe to use to plot the accuracies
    d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
            'acc_entropy':pd.Series(acc_entropy),
            'x_label':pd.Series(x_label)})
    # plots gini accuracies vs entropy accuracies graph, Saves plot diagram in Dataset Folder
    #TempNames are made due to appending of Names, This is used to rename to original value
    TempName = Test_Data['Name']
    TempName2 = Test_Data['Table_Name']
    fig = plt.figure()
    plt.plot('x_label','acc_gini', data=d, label='gini')
    plt.plot('x_label','acc_entropy', data=d, label='entropy')
    plt.xlabel(Test_Data['Table_Name'])
    plt.ylabel('accuracy')

    #Changes Path Based on Parameters
    if(Test_Data['TwoParameters']):
        Test_Data['Table_Name'] = Test_Data['Table_Name']+' Impurity('+str(ImpureLoop)+')'
        Test_Data['Name'] = Test_Data['Name']+'Impurity('+str(ImpureLoop)+')'
        if(Test_Data['MaxDepth'] != None):
            figPath=originalPath+"DecisionTreeIDS/DecisionTreeResults/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Max_Depth/"
        else:
            figPath=originalPath+"DecisionTreeIDS/DecisionTreeResults/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Max_Leaf_Nodes/"
    else:
        figPath=originalPath+"DecisionTreeIDS/DecisionTreeResults/Gini_vs_Entropy_Accuracy_Graphs/Gini_vs_Entropy_Accuracy_Graphs/"
            
    plt.title('Gini vs Entropy - Accuracy vs '+Test_Data['Table_Name'])
    plt.legend()
    #plt.show()
    figPath = figPath+'Gini_vs_Entropy_Feature_'+Test_Data['Name']
    fig.savefig(figPath+'.png')
    Test_Data['Name']=TempName
    Test_Data['Table_Name']=TempName2

#Testing Parameters: Criterion, max_depth, impurity, max_leaf_nodes, 
def Run_All_Test():
    #Loading up Data for Testing
    dataset,attackNames,labelCol = Load_Data_Set()
    #Ran Decision Tree
    for Test_Data in Test_List :
        Run_DecisionTree(dataset=dataset,labelCol=labelCol,attackNames=attackNames, Test_Data=Test_Data,Criterion=Criterion)
        print("Finished: ",Test_Data['Name'])

#Automated to Run all Test in One Run; Just Run Program to start
Run_All_Test()
