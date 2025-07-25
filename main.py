import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns
from sklearn.feature_selection import SelectKBest,mutual_info_classif,mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC,SVR
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from pandas.api.types import is_numeric_dtype,is_string_dtype
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from joblib import dump,load
# taking file input
def isfloat(num):
    try:
        val = float(num)
        return True
    except ValueError:
        return False
    

def fileinput():
    try:
        root = tk.Tk()
        root.withdraw()

        global filepath
        
        filepath = filedialog.askopenfilename(
        title="Select DataSet File",
        filetypes=[("CSV files","*.csv")]
    )
        print("-----------------------------------\nSelected DataSet: ",os.path.basename(filepath))
        global df, original_df
        df = pd.read_csv(filepath)
        original_df = df.copy()
        filterdata()
    except:
        print('File Not Selected! Restart again')
        exit()




def filterdata():
    # df = pd.read_csv(filepath)

    print("-----------------------------------")
    print(f'''Filtered Data record:
[->] Removed Null Values: {(df.isnull().sum().sum())}
[->] Removed Duplicate Rows: {df.duplicated().sum()}
[->] Valid Rows: {len(df)}
[->] Valid Columns: {len(df.columns)}''')
    print("-----------------------------------")

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    findtarget()

def findtarget():
    try:
        print("-----------------------------------")
        for c, i in enumerate(original_df.columns, 1):
            print(c, "->", i)

        global tcolname
        print("-----------------------------------")

        targetcolumn = int(input("Enter Column No. -> "))
        tcolname = original_df.columns[targetcolumn-1]

        print(f"Now Model Will Predict - > {tcolname}")

        global problem_type
        unique_vals = original_df[tcolname].nunique()
        total_vals = len(original_df[tcolname].dropna())
        
        # Better logic for problem type detection
        if is_numeric_dtype(original_df[tcolname]):
            unique_ratio = unique_vals / total_vals
            max_val = original_df[tcolname].max()
            
            if (unique_vals <= 10 and unique_ratio < 0.5 and max_val <= 20 and 
                all(original_df[tcolname].dropna() == original_df[tcolname].dropna().astype(int))):
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            problem_type = 'classification'
        print(f"[✔] Detected Problem Type: {problem_type.capitalize()}")
        print("-----------------------------------")
        modeltraining(original_df.copy(),tcolname)

    except Exception as e:
        print(e)
        findtarget()
    

def makeprediction():
    print("-----------------------------------")
    print("Choose prediction method:")
    print("1 -> Manual Input (Terminal)")
    print("2 -> CSV File Input")
    
    try:
        choice = int(input("Enter choice (1 or 2): "))
        if choice == 1:
            byterminal()
        elif choice == 2:
            byfile()
        else:
            print("Invalid choice! Using manual input.")
            byterminal()
    except ValueError:
        print("Invalid input! Using manual input.")
        byterminal()

def byterminal():
    global allrows
    print("-----------------------------------")
    n_of_pred = int(input("No. Many Predictions : "))
    print(f"Sample below: \n{sampledata.iloc[1]}")
    allrows = []
    c = 0
    
    for i in range(n_of_pred):
        pred_df = {}
        print(f"Enter Data For {c+1} Row: ")
        c += 1
        for col in allcols:
            data = input(f"{col} : ")
            if(data.isnumeric() == True or isfloat(data) == True):
                data = float(data)
            pred_df[col] = data
        allrows.append(pred_df.copy())
    
    makeresult()

def byfile():
    try:
        root = tk.Tk()
        root.withdraw()
        
        filepath = filedialog.askopenfilename(
            title="Select Prediction Data File",
            filetypes=[("CSV files","*.csv")]
        )
        
        if not filepath:
            print("No file selected!")
            return
            
        pred_data = pd.read_csv(filepath)
        print(f"Loaded {len(pred_data)} rows from {os.path.basename(filepath)}")
        
        # Check if all required columns are present
        missing_cols = set(allcols) - set(pred_data.columns)
        if missing_cols:
            print(f"Error: Missing columns in file: {missing_cols}")
            return
        

        pred_data = pred_data[allcols]
        
        global allrows
        allrows = pred_data.to_dict('records')
        
        print("-----------------------------------")
        print("Preview of prediction data:")
        print(pred_data.head())
        makeresult()
        
    except Exception as e:
        print(f"Error reading file: {e}")

def makeresult():
    try:
        pred_df = pd.DataFrame(allrows)
        pred_df = pd.get_dummies(pred_df, drop_first=True)


        for col in dummiescols:
            if col not in pred_df.columns:
                pred_df[col] = 0

        print("-----------------------------------")
        pred_df = pred_df[dummiescols]
        results = bestmodel.predict(pred_df)
        
        for i, pred in enumerate(results, 1):
            print(f"{tcolname} #{i} Result: {pred}")
            if problem_type == 'classification':
                try:
                    prb = bestmodel.predict_proba(pred_df)
                    print(f"Probability: {prb[i-1][1]:.2f}")
                except:
                    print("Probability calculation failed")
        print("-----------------------------------")
        
    except Exception as e:
        print(f"Prediction error: {e}")
        print("Make sure your input data matches the training format.")



def plotgraph():
    if input_data is None or output_data is None or bestmodel is None:
        print("[✘] Missing input/output data or model.")
        return

    print("-----------------------------------")
    print("Available features for X-axis:")
    for i, col in enumerate(input_data.columns, 1):
        print(f"{i} -> {col}")
    
    try:
        choice = int(input("Choose feature number for X-axis: ")) - 1
        if 0 <= choice < len(input_data.columns):
            if isinstance(input_data, pd.DataFrame):
                x = input_data.iloc[:, choice]
                x_column_name = input_data.columns[choice]
            else:
                x = input_data[:, choice]
                x_column_name = f"Feature {choice + 1}"
        else:
            print("Invalid choice, using first feature")
            if isinstance(input_data, pd.DataFrame):
                x = input_data.iloc[:, 0]
                x_column_name = input_data.columns[0]
            else:
                x = input_data[:, 0]
                x_column_name = "Feature 1"
    except (ValueError, IndexError):
        print("Invalid input, using first feature")
        if isinstance(input_data, pd.DataFrame):
            x = input_data.iloc[:, 0]
            x_column_name = input_data.columns[0]
        else:
            x = input_data[:, 0]
            x_column_name = "Feature 1"

    y_actual = output_data
    y_pred = bestmodel.predict(input_data)
    
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))  
    
    plt.plot(x, y_actual, label='Actual data', color='orange', marker='o', linewidth=2, markersize=6)
    plt.plot(x, y_pred, label='Predicted data', color='green', marker='x', linewidth=2, markersize=8)
    
    plt.title(f"{type(bestmodel.named_steps['clf']).__name__} Model Prediction", fontsize=14, fontweight='bold')
    plt.ylabel(f'{tcolname}', fontsize=12)
    plt.xlabel(f'{x_column_name}', fontsize=12)  # Use actual column name
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add R² score to the plot
    if hasattr(bestmodel, 'predict'):
        from sklearn.metrics import r2_score
        r2 = r2_score(y_actual, y_pred)
        plt.text(0.05, 0.95, f'R² Score: {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    print("-----------------------------------")



def modeltraining(df,tcolname):
    global allcols,bestmodel,dummiescols,input_data,output_data,sampledata,problem_type
    df = df[df[tcolname].notna()]
    sampledata = df.drop(tcolname,axis=1)
    input_data = df.drop(tcolname,axis=1)
    allcols = input_data.columns
    output_data = df[tcolname]

    # Drop high-cardinality string columns
    drop_cols = []
    for i in input_data.columns:
        if is_string_dtype(input_data[i]):
            if input_data[i].nunique() > 5:
                drop_cols.append(i)


    input_data.drop(columns=drop_cols, inplace=True)
    input_data = pd.get_dummies(input_data,drop_first=True)
    dummiescols = input_data.columns.tolist()
    # print("[Info] Dropped columns:", drop_cols)
    ogcols = input_data.columns


    df = df[df[tcolname].notna()]


    input_train,input_test,output_train,output_test = train_test_split(input_data,output_data,test_size=0.15,random_state=42)


    featurelength = len(input_data.columns)
    if featurelength <= 2:
        nofcols = 1
    elif featurelength <= 6:
        nofcols = 4
    elif featurelength <= 9:
        nofcols = 7
    else:
        nofcols = 9

    
    if problem_type == 'classification':
        kbest = mutual_info_classif
    else:
        kbest = mutual_info_regression

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=kbest,k=nofcols)),
        ('clf', LogisticRegression())
    ])

    param_grids_classification = [
        {
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': [10,20,50],
            'clf__max_depth': [None, 10, 20]
        },
        {
            'clf': [LogisticRegression()],
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__max_iter': [100, 200, 500]
        },
        {
            'clf': [SVC()],
            'clf__C': [0.1, 1,10],
            'clf__kernel': ['linear', 'rbf']
        },
        {
            'clf': [BernoulliNB()],
            'clf__alpha': [0.1, 1],
            'clf__binarize': [0.0, 0.5]
        }
    ]

    param_grids_regression = [
        {
            'clf': [RandomForestRegressor(random_state=42)],
            'clf__n_estimators': [5,10,50,100,200],
            'clf__max_depth': [None,1,10,20,50]
        },
        {
            'clf': [LinearRegression()]
        },
        {
            'clf': [SVR()],
            'clf__C': [0.1, 1],
            'clf__kernel': ['linear', 'rbf']
        }
    ]

    if(problem_type=='classification'):
        gridata = param_grids_classification
    else:
        gridata = param_grids_regression


    
    if problem_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    



    grid = GridSearchCV(pipe,param_grid=gridata,cv=cv_strategy,error_score='raise')
    try:
        grid.fit(input_train,output_train)
    except Exception as e:
        print(e)




    bestmodel = grid.best_estimator_
    predicted_data = bestmodel.predict(input_test)
    print(f"Datased Trained On: {type(bestmodel.named_steps['clf']).__name__}")
    if problem_type=='classification':
        print(f"Accuracy Score: {accuracy_score(output_test,predicted_data)}")
    else:
        print(f"R² Score: {r2_score(output_test, predicted_data):.4f}")

    try:
        selector = bestmodel.named_steps['select']
        selectedcols = ogcols[selector.get_support()].tolist()
        # print("Selected top", selector.k, "Columns.")
        # print("Names:: ",selectedcols)

        # print("Selected Features are: ",selectedcols)
    except Exception as e:
        print(e)
    
    try:
        name = input("Enter name for your model:: ")
        os.makedirs('savedmodels',exist_ok=True)
        bundle = {
    "model": bestmodel,
    "dummiescols": dummiescols,
    "allcols": allcols,
    "input_data": input_data,
    "output_data": output_data,
    "tcolname": tcolname,
    "sampledata": sampledata,
    "problem_type": problem_type
    }
        dump(bundle,f"savedmodels/{name}.joblib")
        print(f"{name} Has been saved ✅")

    except Exception as e:
        print(e)


def modelmenu():
    while(1):
        choice = (input(f'''
--------------------------
1-> Make Prediction
2-> Generate Graph
3-> Back to Main menu
4-> Terminate session
[-]: '''))
        if choice=="1":
            makeprediction()
        elif choice=="2":
            plotgraph()
        elif(choice=="3"):
            mainmenu()
        elif(choice=="4"):
            print("Terminating..")
            exit()
        else:
            print("Wrong choice")
        

def usingtrainedmodels():
    global bestmodel,input_data,output_data,allcols,dummiescols,tcolname,sampledata
    print("-----------------------------------")
    print("Available Trained Models 👇")
    models = os.listdir("savedmodels")
    
    if not models:
        print("[!] No saved models found.")
        return
    
    for i, model_name in enumerate(models, 1):
        print(f"{i} -> {model_name}")
    
    try:
        choice = int(input("\nEnter the number of the model to load: "))
        if 1 <= choice <= len(models):
            selected_model_path = os.path.join("savedmodels", models[choice - 1])
            bundle = load(f"{selected_model_path}")
            bestmodel = bundle["model"]
            dummiescols = bundle["dummiescols"]
            input_data = bundle["input_data"]
            output_data = bundle["output_data"]
            allcols = bundle["allcols"]
            tcolname = bundle["tcolname"]
            sampledata = bundle["sampledata"]
            problem_type = bundle["problem_type"]
            print(f"[✔] Model '{models[choice - 1]}' loaded successfully.")
            modelmenu()
        else:
            print("[!] Invalid choice. Please select a valid number.")
    except ValueError:
        print("[!] Invalid input. Please enter a number.")



def mainmenu():
    while(1):
        choice = (input(f'''
----------------------
1-> Train a new model
2-> Use trained model
3-> Exit
[->]: '''))
        if(choice=="1"):
            fileinput()
        elif(choice=="2"):
            usingtrainedmodels()
        elif(choice=="3"):
            break
        else:
            print("WRONG INPUT TRY AGAIN")


print(f'''🄼🄻 🄲🅁🄰🄵🅃🄴🅁 v2'''.center(50))
mainmenu()
