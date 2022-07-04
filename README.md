# JAK_ML

1. Data processing
   Data were extracted from pubchem, bindingDB, and ChemBL as csv files. Empty files deleted. Aim is to build separate model for JAK1, JAK2, JAK3, and TYK2 to predict inhibitors.
   raw data was handled: 
     * Active: 1
     * Inactive: 0
     * Unspecifed: -1
     * Inconclusive: deleted, not in file anymore

   CID (Compound ID) was replaced as smiles. 

   Further, the unspecifed -1 was merged with Inactive 0 as there were much fewer inactive drugs. 

   Same assay data were merged into a large csv file. 

   Since data were extracted from bunch of assays, delete duplicate is needed 

After training SVM, RF, RGBoost (maybe Bert CNN, Graph based models also applicable), SHAP values were calculated to evaluate the importance of features, which could be converted to substructures of the compound. Thus could help alleviate black box in machine learning. By doing explainable ML, we could figure out which substructures could lead to positive output.   

FLT3 were also extracted from pubchem. Since nonspecific targeting to FLT3 is related with GI toxicity to JAK inhibitors, we may also build a model to predict compounds on FLT3 inhibition, and would like to avoid FLT3 inhibition to lower GI toxicity for drugs targeting IBD (GI tract diseases)
