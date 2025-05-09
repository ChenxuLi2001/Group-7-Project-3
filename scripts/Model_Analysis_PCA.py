
def run_Model_Analysis_PCA():
    #import modules
    #for loading and cleaning
    import pandas as pd
    from IPython.display import display

    #for classification (only additional modules)
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay

    #for PCA analysis (only additional modules)
    import numpy as np
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    #load the merged dataset
    try:
      data = pd.read_csv('FinalCleanedMergedResearchOutputs.csv')
    except Exception as e:
      print('Error loading file:',e)
    else:
      print('File loaded successfully')
      print(data.shape)
      #display(data.head())

      #get null information
      null_summary = pd.DataFrame({
        'null_count':  data.isnull().sum(),
        'null_percent': (data.isnull().mean() * 100).round(2)
      })
      display(null_summary)

    #Create a new dataframe and drop any columns that have a null percentage over 5
    try:
      missing_pct = data.isna().mean().sort_values(ascending=False)
      tokeep = missing_pct[missing_pct < 0.05].index
      cleaned = data[tokeep].copy()
      cleaned['ProjID']     = pd.to_numeric(cleaned['ProjID'], errors='coerce').astype('Int64')
      cleaned['OutputYear'] = pd.to_numeric(cleaned['OutputYear'], errors='coerce').astype('Int64')
    except Exception as e:
      print('Error cleaning data:',e)
    else:
      print('Data cleaned successfully')
      print(cleaned.shape)
      display(cleaned.head())

    #Classification based on outputstatus
    try:
      #create new dataframe without NaN values in OutputStatus
      Outputstatus = cleaned.dropna(subset=['OutputStatus']).copy()

      #Convert the target to categorical and encode
      Outputstatus['OutputStatus'] = Outputstatus['OutputStatus'].astype('category')
      le = LabelEncoder()
      Outputstatus['y_multi'] = le.fit_transform(Outputstatus['OutputStatus'])
      mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      print("Label encoding:", mapping)

      #Define features & target
      features = ['OutputTitle','ProjectStatus','OutputYear','ProjectRDC','OutputType']
      X = Outputstatus[features]
      y = Outputstatus['y_multi']

      #impute missing values in features selected
      title_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('squeeze', FunctionTransformer(lambda X: X.ravel(), validate=False)),
        ('tfidf',   TfidfVectorizer(max_features=200, stop_words='english'))
      ])
      numeric_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='median')),
          ('scaler', StandardScaler())
      ])
      categorical_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
          ('onehot', OneHotEncoder(handle_unknown='ignore'))
      ])
      preprocessor = ColumnTransformer([
              ('title', title_pipe, ['OutputTitle']),
              ('num', numeric_transformer, ['OutputYear']),
              ('cat', categorical_transformer, ['ProjectStatus','ProjectRDC','OutputType'])
          ])

      #set weights for output status to account for low FC data
      cw = {
        mapping['FC']: 50,
        mapping['PB']: 1,
        mapping['UP']: 1
      }

      #create classification model
      clf_multi = Pipeline([
        ('prep', preprocessor),
        ('svc', SVC(
            kernel='rbf',
            class_weight=cw,
            probability=True,    # required for AUC scoring
            random_state=42
        ))
      ])

      #paraneters for tuning model
      param_grid = {
        'svc__C':     [0.01, 0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1],
        # you can also tune kernel, degree (for poly), etc.
      }

      #run gridsearch to select best parameters
      grid = GridSearchCV(
        estimator=clf_multi,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc_ovo_weighted', # or 'accuracy'
        n_jobs=-1,                      # parallelize across cores
        verbose=1                       # print progress messages
      )

      grid.fit(X, y)

      print("Best parameters:", grid.best_params_)
      print("Best CV AUC:    ", grid.best_score_)


    except Exception as e:
      print('Error cleaning data:',e)
    else:
      print("Classification testing done")
      # fit on all data and predict and print confusion matrix
      try:
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X)
        print(classification_report(y, y_pred, target_names=le.classes_, zero_division=0))
        ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=le.classes_)


      except Exception as e:
        print('Error cleaning data:',e)
      else:
        print("Classification completed")

    #PCA insights
    try:

      #get data
      df = cleaned[['OutputYear','ProjectStatus','ProjectRDC']].copy()

      #drop missing rows
      df = df.dropna()
      df.reset_index(drop=True, inplace=True)
      # Normalize ProjectStatus to Title Case
      df['ProjectStatus'] = df['ProjectStatus'].str.strip().str.title()


      # One‑hot encode categorical cols
      enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
      cat_matrix = enc.fit_transform(df[['ProjectStatus','ProjectRDC']])

      # Stack with OutputYear and scale
      X = np.hstack([df[['OutputYear']].values, cat_matrix])
      X_scaled = StandardScaler().fit_transform(X)

      # Fit PCA
      pca = PCA()
      pca.fit(X_scaled)
      evr = pca.explained_variance_ratio_

      # Scree plot
      plt.figure()
      plt.plot(np.arange(1, len(evr)+1), evr, 'o-')
      plt.xlabel('Principal Component')
      plt.ylabel('Explained Variance Ratio')
      plt.title('PCA Scree Plot')
      plt.show()

      # Loadings for PC1 & PC2
      feature_names = ['OutputYear'] + enc.get_feature_names_out(['ProjectStatus','ProjectRDC']).tolist()
      loadings = pd.DataFrame(pca.components_[:2].T, index=feature_names, columns=['PC1','PC2'])

    except Exception as e:
      print("Error with PCA", e)

    else:
      print("Top drivers of PC1:")
      print(loadings['PC1'].abs().sort_values(ascending=False).head(10))
      print("\nTop drivers of PC2:")
      print(loadings['PC2'].abs().sort_values(ascending=False).head(10))

    try:
      #remove pages and fill na
      Pages = data[['OutputPages']].copy()
      Pages["OutputPages"] = Pages["OutputPages"].fillna("missing")

      #Use regex to get those with a page range
      dash = r'[-–—‑]'  # hyphen, en‑dash, em‑dash, non‑breaking hyphen
      Pages['IsRange'] = Pages['OutputPages'].str.contains(dash, na=False)

      #Pull out first and last page numbers
      rex = Pages.loc[Pages['IsRange'], 'OutputPages'] \
                .str.extract(r'(?P<start>\d+)\D*[-–—‑]\D*(?P<end>\d+)')
      rex = rex.fillna("1") #make all wrong page formats as 1
      rex = rex.astype(int) #set to integers

      #add page count to pages
      Pages.loc[Pages['IsRange'], 'PageCount'] = (
          rex['end'].astype(int)
          - rex['start'].astype(int)
          + 1
      )

      #Fill in the rest as 1 page unless it is missing then no pages and change to int
      Pages['PageCount'] = np.where(
          Pages['IsRange'],
            Pages['PageCount'],
          np.where(
            Pages['OutputPages'] != "missing",
            1,
            0
          )
      )
      Pages["PageCount"] = Pages["PageCount"].astype(int)

      #For numerical PCA analysis
      #create new copy and and Page count
      newdf = data.copy()
      newdf['PageCount'] = Pages['PageCount']
      #get columns with numberical values and drop NaN rows
      numericalcols = newdf.select_dtypes(include=['int64', 'float64']).columns
      numdf = newdf[numericalcols].copy()
      numdf = numdf.dropna()
      #drop ProjID
      numdf = numdf.drop(columns=['ProjID'])
      numdf["ProjectDuration"] = numdf['ProjectYearEnded']-numdf['ProjectYearStarted']
      #numdf = numdf.drop(columns=['ProjectYearStarted','ProjectYearEnded'])

      #get PCA insights
      pca = PCA()
      pca.fit(numdf)
      evr = pca.explained_variance_ratio_

      plt.figure()
      plt.plot(np.arange(1, len(evr)+1), evr, 'o-')
      plt.xlabel('Principal Component')
      plt.ylabel('Explained Variance Ratio')
      plt.title('PCA Scree Plot')
      plt.show()

      pca = PCA()
      pcs = pca.fit_transform(numdf)
      evr = pca.explained_variance_ratio_

      # 2) Build loadings DataFrame
      loadings = pd.DataFrame(
          pca.components_.T,
          index=numdf.columns,
          columns=[f'PC{i}' for i in range(1, len(evr)+1)]
      )

    except Exception as e:
      print("Error with PCA analysis",e)

    else:
      #Top 3 drivers for PC1, PC2, PC3
      for pc in ['PC1','PC2','PC3']:
          top = loadings[pc].abs().nlargest(3).index
          print(f"\nTop drivers of {pc}:")
          display(loadings.loc[top, [pc]])

if __name__ == "__main__":
    run_Model_Analysis_PCA()
